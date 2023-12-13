
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Tuple, Dict, List, Union
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler


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


def collate_sampler_data(data: list):
    outputs = defaultdict(list)
    for complex_data, chain_key in data:

        print(chain_key, '->')
        chain_key_tuple = tuple(chain_key.split('-')[1:])
        curr_chain_sequence = complex_data[chain_key_tuple]['polymer_seq']

        for chain_tup, chain_data in complex_data.items():
            print(chain_data)

            identical_chains = chain_data['polymer_seq'] == curr_chain_sequence
            if identical_chains:
                chain_mask = torch.ones(chain_data['size'], dtype=torch.bool)
            else:
                chain_mask = torch.zeros(chain_data['size'], dtype=torch.bool)
            
            
            raise NotImplementedError


class ClusteredDatasetSampler(Sampler):
    """
    Samples a single protein complex from precomputed mmseqs clusters.
    Ensures samples drawn evenly by sampling first from sequence clusters, then by pdb_code, then by assembly and chain.
    Iteration returns batched indices for use in UnclusteredProteinChainDataset.
    Pass to a DataLoader as a batch_sampler.
    """
    def __init__(self, dataset, params):
        # The unclustered dataset where each complex/assembly is a single index.
        self.dataset = dataset
        self.batch_size = params['batch_size']

        # Load the cluster data.
        print("Loading sequence clusters.")
        df = pd.read_csv(os.path.join(params['clustering_output_path'], f"{params['clustering_output_prefix']}_cluster.tsv"), sep='\t', header=None)
        df.columns = ['cluster_representative', 'chain']

        # Maps a given pdb_code+chain to its representative cluster.
        self.chain_to_cluster = create_chain_to_cluster_mapping(df, params)

        # Maps sequence cluster to number of chains.
        self.cluster_to_chains = invert_dict(self.chain_to_cluster)

        # Sample the first epoch.
        self.curr_samples = []
        self.sample_clusters()

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
    
        # Yield the indexes in the order of the sorted sizes.
        outputs = []
        for batch in torch.chunk(curr_samples_tensor[torch.argsort(sizes)], len(self)):
            outputs.append(batch.tolist())
        np.random.shuffle(outputs)
    
        for batch in outputs:
            yield batch

        # Resample for the next epoch.
        self.sample_clusters()


class UnclusteredProteinChainDataset(Dataset):
    """
    Dataset where every pdb_assembly-segment-chain is a separate index.
    """
    def __init__(self, params):
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
        return self.pdb_code_to_complex_data[pdb_code], chain_key

    
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