#!/usr/bin/env python3

import os
import torch
from tqdm import tqdm

from utils.dataset import ClusteredDatasetSampler, UnclusteredProteinChainDataset, collate_sampler_data
from utils.build_rotamers import RotamerBuilder
from utils.model import Reinforcemer
from torch.utils.data import DataLoader 

def main(params):
    protein_dataset = UnclusteredProteinChainDataset(params)
    sampler = ClusteredDatasetSampler(protein_dataset, params)
    dataloader = DataLoader(protein_dataset, batch_sampler=sampler, collate_fn=collate_sampler_data, num_workers=2, persistent_workers=True)

    rotamer_builder = RotamerBuilder()
    model = Reinforcemer()

    for epoch_num in range(params['num_epochs']):
        for batch in tqdm(dataloader, total=len(dataloader), leave=False, desc=f'Training Epoch {epoch_num}'):
            pass
    raise NotImplementedError

if __name__ == "__main__":
    params = {
        'debug': (debug := True),
        'num_epochs': 100,
        'batch_size': 10_000,
        'sample_randomly': True,
        'device': 'cuda:6',
        'dataset_path': '/scratch/bfry/torch_bioasmb_dataset' + ('/aa' if debug else ''),
        'clustering_output_prefix': 'torch_bioas_cluster30',
        'clustering_output_path': (output_path := '/scratch/bfry/bioasmb_dataset_sequence_clustering/'),
    }
    if params['debug']:
        print('Running in debug mode!')
    main(params)