
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Tuple, Dict, Optional
from collections import defaultdict
from sklearn.model_selection import train_test_split

import torch
from torch_cluster import knn_graph
from torch.utils.data import Dataset, Sampler
from utils.constants import MAX_PEPTIDE_LENGTH, NUM_CB_ATOMS_FOR_BURIAL
from dataclasses import dataclass


@dataclass
class BatchData():
    """
    Dataclass storing all the data for a batch of proteins for input to model.
    """
    chain_mask: torch.Tensor # True if residue is NOT being trained over.
    extra_atom_contact_mask: torch.Tensor # True if residue is in contact with extra atoms.
    sequence_indices: torch.Tensor
    chi_angles: torch.Tensor
    backbone_coords: torch.Tensor
    residue_buried_mask: torch.Tensor
    batch_indices: torch.Tensor

    edge_index: Optional[torch.Tensor] = None
    edge_distance: Optional[torch.Tensor] = None

    def construct_graph(self, training_noise: float) -> None:
        """
        Computes a KNN graph using CA coordinates and distances between all pairs of atoms.
        Stores the edge_index and edge_distance tensors in the BatchData object.
        """
        self.edge_index = knn_graph(self.backbone_coords[:, 1], k=10, batch=self.batch_indices, loop=True)
        noised_backbone_coords = self.backbone_coords + (training_noise * torch.randn_like(self.backbone_coords))
        self.edge_distance = torch.cdist(noised_backbone_coords[self.edge_index[0]], noised_backbone_coords[self.edge_index[1]]).flatten(start_dim=1)
    
    def to_device(self, device: torch.device) -> None:
        """
        Moves all tensors in the BatchData object to the specified device.
        """
        for k,v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                self.__dict__[k] = v.clone().to(device)


def get_list_of_all_paths(path: str) -> list:
    """
    Recursively get a list of all paths of pytorch files in a directory.
    """
    all_paths = []
    for subdir_or_files in os.listdir(path):
        path_to_subdir_or_file = os.path.join(path, subdir_or_files)
        if path_to_subdir_or_file.endswith('.pt'):
            all_paths.append(path_to_subdir_or_file)
        elif os.path.isdir(path_to_subdir_or_file):
            all_paths.extend(get_list_of_all_paths(path_to_subdir_or_file))
    return all_paths


def create_chain_to_cluster_mapping(df: pd.DataFrame, params: dict) -> Dict[str, str]:
    chain_to_cluster = {}
    for _, row in df.iterrows():
        chain = row['chain']
        if params['debug']:
            if not chain.split('_')[0][1:3] == params['dataset_path'].rsplit('/', 1)[1]:
                continue
        cluster_representative = row['cluster_representative']
        chain_to_cluster[chain] = cluster_representative
    return chain_to_cluster


def invert_dict(d: dict) -> dict:
    clusters = defaultdict(list)
    for k, v in d.items():
        clusters[v].append(k)
    return dict(clusters)


def chain_list_to_protein_chain_dict(chain_list: list) -> dict:
    """
    Takes a list of bioassemblies+segment+chains and returns a dictionary 
    mapping pdb code to a list of assemblies and chains in a given sequence cluster.
    """

    bioasmb_list = defaultdict(list)
    for chain in chain_list:
        pdb_code, asmb_chain_id = chain.split('_')
        bioasmb_list[pdb_code].append(asmb_chain_id)

    return dict(bioasmb_list)


def get_complex_len(complex_data: dict) -> int:
    return sum([x['size'] for x in complex_data.values()])


def collate_sampler_data(data: list) -> BatchData:
    all_batch_data = defaultdict(list)
    for idx, (complex_data, chain_key) in enumerate(data):

        # Get the sequence of the chain that we sampled the assembly with from cluster.
        chain_key_tuple = tuple(chain_key.split('-')[1:])
        curr_chain_sequence = complex_data[chain_key_tuple]['polymer_seq']

        # Loop over the remaining chains and add them to the batch.
        for _, chain_data in complex_data.items():

            # If the seuqence of the current chain is the same as the chain 
            #   we sampled from the cluster, set the chain mask to all ones.
            #   Also just provide rotamers for anything less than MAX_PEPTIDE_LENGTH.
            identical_chains = chain_data['polymer_seq'] == curr_chain_sequence
            if identical_chains or chain_data['size'] <= MAX_PEPTIDE_LENGTH:
                chain_mask = torch.ones(chain_data['size'], dtype=torch.bool)
            else:
                chain_mask = torch.zeros(chain_data['size'], dtype=torch.bool)

            # Extract the rest of the chain data.
            extra_atom_contact_mask = chain_data['extra_atom_contact_mask']
            sequence_indices = chain_data['sequence_indices']
            chi_angles = chain_data['chi_angles']
            backbone_coords = chain_data['backbone_coords']
            residue_buried_mask = chain_data['residue_cb_counts'] > NUM_CB_ATOMS_FOR_BURIAL
            batch_indices = torch.full((backbone_coords.shape[0],), idx, dtype=torch.long)

            all_batch_data['chain_mask'].append(chain_mask)
            all_batch_data['extra_atom_contact_mask'].append(extra_atom_contact_mask)
            all_batch_data['sequence_indices'].append(sequence_indices.long())
            all_batch_data['chi_angles'].append(chi_angles)
            all_batch_data['backbone_coords'].append(backbone_coords)
            all_batch_data['residue_buried_mask'].append(residue_buried_mask)
            all_batch_data['batch_indices'].append(batch_indices)

    # Concatenate all the data in the batch dimension.
    outputs = {}
    for i,j in all_batch_data.items():
        outputs[i] = torch.cat(j, dim=0)
    
    # Create a BatchData object and compute the KNN graph.
    output_batch_data = BatchData(**outputs)

    return output_batch_data

def generate_cluster_splits(params: dict, cluster_to_chains: dict, cluster_split_seed: int) -> None:
    """
    Generates train and test splits for the mmseqs clusters.
    """
    all_cluster_keys = np.array(list(cluster_to_chains.keys()))
    train_data, test_data = train_test_split(all_cluster_keys, test_size=0.2, random_state=cluster_split_seed)
    torch.save(set(train_data), params['train_splits_path'])
    torch.save(set(test_data), params['test_splits_path'])


class UnclusteredProteinChainDataset(Dataset):
    """
    Dataset where every pdb_assembly-segment-chain is a separate index.
    """
    def __init__(self, params, transform=None):
        self.transform = transform
        self.pdb_code_to_complex_data = {} # Maps from pdb_code to protein complex/bioassembly data.
        self.chain_key_to_index = {} # Maps from unique index to chain_key
        self.index_to_complex_size = {}
        idx = 0
        for path_to_data in tqdm(get_list_of_all_paths(params['dataset_path']), desc='Loading protein dataset.'):
            pdb_prefix = path_to_data.rsplit('/', 1)[1].replace('.pt', '')

            # Load the protein complex from disk, store with pdb_prefix as key.
            protein_data = torch.load(path_to_data)
            self.pdb_code_to_complex_data[pdb_prefix] = protein_data

            # Loop over chains and assign a unique index to chain_key
            for chain_key, chain_data in protein_data.items():
                chain_key = '-'.join([pdb_prefix] + list(chain_key))
                self.chain_key_to_index[chain_key] = idx
                self.index_to_complex_size[idx] = get_complex_len(protein_data)
                idx += 1
        self.index_to_chain_key = {x: y for y,x in self.chain_key_to_index.items()}

    def __len__(self) -> int:
        return len(self.chain_key_to_index)

    def __getitem__(self, index: int) -> Tuple[dict, str]:
        # Take indexes unique to chain and return the complex data for that chain and the chain key.
        chain_key = self.index_to_chain_key[index]
        pdb_code = chain_key.split('-')[0]
        output_data = self.pdb_code_to_complex_data[pdb_code]
        if self.transform:
            output_data = self.transform(output_data)
        return output_data, chain_key


class ProteinAssemblyDataset(Dataset):
    """
    Dataset where every biological assembly is a separate index.
    """
    def __init__(self, params):
        self.data = {}
        for path_to_data in tqdm(get_list_of_all_paths(params['dataset_path']), desc='Loading protein dataset.'):
            pdb_prefix = path_to_data.rsplit('/', 1)[1]
            self.data[pdb_prefix] = torch.load(path_to_data)
        self.index_to_key = list(self.data.keys())

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[dict, str]:
        pdb_code = self.index_to_key[index]
        return self.data[pdb_code], pdb_code
    

class ClusteredDatasetSampler(Sampler):
    """
    Samples a single protein complex from precomputed mmseqs clusters.
    Ensures samples drawn evenly by sampling first from sequence clusters, then by pdb_code, then by assembly and chain.
    Iteration returns batched indices for use in UnclusteredProteinChainDataset.
    Pass to a DataLoader as a batch_sampler.
    """
    def __init__(self, dataset: UnclusteredProteinChainDataset, params: dict, is_test_dataset_sampler: bool, cluster_split_seed: int = 0):
        """
        Takes a torch dataset, param dictionary, adn a boolean indicating whether to sample from the train or test clusters.
        """
        # The unclustered dataset where each complex/assembly is a single index.
        self.dataset = dataset
        self.batch_size = params['batch_size']
        self.sample_randomly = params['sample_randomly']
        self.max_protein_length = params['max_protein_size']

        # Load the cluster data.
        print("Loading sequence clusters.")
        df = pd.read_csv(os.path.join(params['clustering_output_path'], f"{params['clustering_output_prefix']}_cluster.tsv"), sep='\t', header=None, names=['cluster_representative', 'chain'])

        # Maps a given pdb_code+chain to its representative cluster.
        self.chain_to_cluster = create_chain_to_cluster_mapping(df, params)

        # Maps sequence cluster to number of chains.
        self.cluster_to_chains = invert_dict(self.chain_to_cluster)

        # Generate random train and test cluster splits if necessary then load.
        if not (os.path.exists(params['train_splits_path']) and os.path.exists(params['test_splits_path'])):
            generate_cluster_splits(params, self.cluster_to_chains, cluster_split_seed)

        # Load relevant pickled sets of cluster keys, filter for train/test as necessary.
        self.train_split_clusters = torch.load(params['train_splits_path'])
        self.test_split_clusters = torch.load(params['test_splits_path'])
        self.cluster_to_chains = self.filter_clusters(is_test_dataset_sampler)

        # Sample the first epoch.
        self.curr_samples = []
        self.sample_clusters()
    
    def filter_clusters(self, is_test_dataset_sampler: bool) -> dict:
        """
        Filter clusters based on the given dataset sampler.
            Parameters:
            - is_test_dataset_sampler (bool): True if the dataset sampler is for the test dataset, False otherwise.

            Returns:
            - dict: A dictionary containing the filtered clusters.
        """

        if self.cluster_to_chains is None:
            raise NotImplementedError("Unreachable.")

        if is_test_dataset_sampler:
            curr_cluster_set = self.test_split_clusters
        else:
            curr_cluster_set = self.train_split_clusters
        
        # Filter the clusters for train or test set.
        output = {k: v for k,v in self.cluster_to_chains.items() if k in curr_cluster_set}

        # If we don't have a max protein length, return the output.
        if self.max_protein_length is None:
            return output
        
        # Drop things that are longer than the max protein length.
        filtered_output = defaultdict(list)
        for cluster_rep, cluster_list in output.items():
            for chain in cluster_list:
                chain_len = self.dataset.index_to_complex_size[self.dataset.chain_key_to_index[chain]]
                if chain_len <= self.max_protein_length:
                    filtered_output[cluster_rep].append(chain)
        return filtered_output
    

    def get_curr_sample_len(self) -> int:
        """
        Returns the total length of the current samples in the dataset.
        """
        return sum(self.dataset.index_to_complex_size[x] for x in self.curr_samples)

    def __len__(self) -> int:
        """
        Returns number of batches in the current epoch.
        """
        return (self.get_curr_sample_len() + self.batch_size - 1) // self.batch_size
    
    def sample_clusters(self) -> None:
        """
        Randomly samples clusters from the dataset for the next epoch.
        Updates the self.curr_samples list with new samples.
        """
        self.curr_samples = []
        # Loop over mmseqs cluster and list of chains for that cluster.
        for cluster, chains in self.cluster_to_chains.items():
            # Convert list of all chains/pdbs/assemblies to a dictionary mapping pdb code to a 
            # list of assemblies and chains in the current cluster.
            pdb_to_assembly_chains_map = chain_list_to_protein_chain_dict(chains)

            # Sample from the PDBs with the desired chain cluster.
            sampled_pdb = np.random.choice(list(pdb_to_assembly_chains_map.keys()))

            # Given the PDB to sample from sample an assembly and chain for training.
            sampled_assembly_and_chains = np.random.choice(pdb_to_assembly_chains_map[sampled_pdb])
    
            # Reform the string representation of the sampled pdb_assembly-seg-chain.
            chain_key = '_'.join([sampled_pdb, sampled_assembly_and_chains])

            # Yield the index of the sampled pdb_assembly-seg-chain.
            self.curr_samples.append(self.dataset.chain_key_to_index[chain_key])
        
    def __iter__(self):
        """
        Batches by size inspired by:
        https://pytorch.org/docs/stable/data.html#torch.utils.data.Sampler:~:text=%3E%3E%3E%20class%20AccedingSequenceLengthBatchSampler,.tolist()
        """
        # Sort the samples by size.
        curr_samples_tensor = torch.tensor(self.curr_samples)
        sizes = torch.tensor([self.dataset.index_to_complex_size[x] for x in self.curr_samples])
        size_sort_indices = torch.argsort(sizes)

        # if self.max_protein_length is not None:
        #     size_mask = sizes <= self.max_protein_length

        #     sizes = sizes[size_mask]
        #     curr_samples_tensor = curr_samples_tensor[size_mask]
        #     size_sort_indices = torch.argsort(sizes)

        # iterate through the samples in order of size, create batches of size batch_size.
        index = 0
        debug_sizes = []
        outputs, curr_list_sample_indices, curr_list_sizes = [], [], []
        while index < len(size_sort_indices):
            while sum(curr_list_sizes) < self.batch_size and index < len(size_sort_indices):
                # Get current sample index and size.
                curr_size_sort_index = size_sort_indices[index]
                curr_sample_index = curr_samples_tensor[curr_size_sort_index].item()
                curr_size = sizes[curr_size_sort_index].item()
                # Add to the current batch.
                curr_list_sample_indices.append(curr_sample_index)
                curr_list_sizes.append(curr_size)
                index += 1
            # Add the current batch to the list of batches.
            outputs.append(curr_list_sample_indices)
            debug_sizes.append(sum(curr_list_sizes))
            # Reset for next batch.
            curr_list_sizes = []
            curr_list_sample_indices = []

        # Shuffle the batches.
        if self.sample_randomly:
            np.random.shuffle(outputs)

        # Sanity check that we have the correct number of samples after iteration.
        assert(sum(debug_sizes) == sizes.sum().item())

        # Yield the batches we created.
        for batch in outputs:
            yield batch

        # Resample for the next epoch.
        self.sample_clusters()
