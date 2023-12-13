#!/usr/bin/env python3

import os
import torch

from utils.dataset import ClusteredDatasetSampler, UnclusteredProteinChainDataset, collate_sampler_data
from torch.utils.data import DataLoader 

def main():
    protein_dataset = UnclusteredProteinChainDataset(params)
    sampler = ClusteredDatasetSampler(protein_dataset, params)
    dataloader = DataLoader(protein_dataset, batch_sampler=sampler, collate_fn=collate_sampler_data)
    for data in dataloader:
        for i in data.items():
            print(i[0], i[1].shape)
        print()
    raise NotImplementedError

if __name__ == "__main__":
    params = {
        'batch_size': 10_000,
        'sample_randomly': True,
        'debug': (debug := True),
        'dataset_path': '/scratch/bfry/torch_bioasmb_dataset' + '/aa' if debug else '',
        'clustering_output_prefix': 'torch_bioas_cluster30',
        'clustering_output_path': (output_path := '/scratch/bfry/bioasmb_dataset_sequence_clustering/'),
    }
    main()