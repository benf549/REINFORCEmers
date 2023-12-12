
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Tuple, Dict
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler

def create_chain_to_cluster_mapping(df: pd.DataFrame) -> Dict[str, str]:
    chain_to_cluster = {}
    for _, row in df.iterrows():
        chain = row['chain']
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

class ClusteredDatasetSampler(Sampler):
    def __init__(self, dataset, params):

        # The unclustered dataset where each complex/assembly is a single index.
        self.dataset = dataset

        # Load the cluster data.
        df = pd.read_csv(os.path.join(params['clustering_output_path'], f"{params['clustering_output_prefix']}_cluster.tsv"), sep='\t', header=None)
        df.columns = ['cluster_representative', 'chain']

        # Maps a given pdb_code+chain to its representative cluster.
        self.chain_to_cluster = create_chain_to_cluster_mapping(df)

        # Maps sequence cluster to number of chains.
        self.cluster_to_chains = invert_dict(self.chain_to_cluster)

    def __len__(self):
        return len(self.cluster_to_chains)

    def __iter__(self):
        # Loop over mmseqs cluster and list of chains for that cluster.
        for cluster, chains in self.cluster_to_chains.items():
            pdb_to_assembly_chains_map = chain_list_to_protein_chain_dict(chains)
            sampled_pdb = np.random.choice(list(pdb_to_assembly_chains_map.keys()))
            sampled_assembly_and_chains = np.random.choice(pdb_to_assembly_chains_map[sampled_pdb])
            sampled_pdb = '_'.join([sampled_pdb, sampled_assembly_and_chains])
            print(cluster, '->', sampled_pdb)

        raise NotImplementedError


class UnclusteredProteinDataset(Dataset):
    def __init__(self, params):
        self.data = {}
        for data in tqdm(os.listdir(params['dataset_path']), desc='Loading protein dataset.'):
            path_to_data = os.path.join(params['dataset_path'], data)
            pdb_prefix = path_to_data.rsplit('/', 1)[1]
            self.data[pdb_prefix] = torch.load(path_to_data)
        self.index_to_key = list(self.data.keys())

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[dict, str]:
        pdb_code = self.index_to_key[index]
        return self.data[pdb_code], pdb_code
    
    def get_item_with_pdb_code(self, pdb_code: str) -> Tuple[dict, str]:
        return self.data[pdb_code]
