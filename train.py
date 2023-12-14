#!/usr/bin/env python3

import os
import torch
from tqdm import tqdm

from utils.dataset import ClusteredDatasetSampler, UnclusteredProteinChainDataset, collate_sampler_data
from utils.build_rotamers import RotamerBuilder
from utils.model import ReinforcemerRepacker
from torch.utils.data import DataLoader 

def main(params):
    protein_dataset = UnclusteredProteinChainDataset(params)
    sampler = ClusteredDatasetSampler(protein_dataset, params)
    dataloader = DataLoader(protein_dataset, batch_sampler=sampler, collate_fn=collate_sampler_data, num_workers=params['num_workers'], persistent_workers=True)

    model = ReinforcemerRepacker(**params['model_params'])
    optimizer = torch.optim.Adam(model.parameters(), params['learning_rate'])

    for epoch_num in range(params['num_epochs']):
        for batch in tqdm(dataloader, total=len(dataloader), leave=False, desc=f'Training Epoch {epoch_num}'):
            optimizer.zero_grad()

            out = model(batch)

            raise NotImplementedError
            optimizer.step()
        
        break

if __name__ == "__main__":
    params = {
        'debug': (debug := True),
        'num_workers': 1,
        'num_epochs': 100,
        'batch_size': 10_000,
        'learning_rate': 1e-4,
        'sample_randomly': True,

        'model_params': {
            'dropout': 0.1,
            'node_embedding_dim': 128,
            'edge_embedding_dim': 128,
            'num_encoder_layers': 3,
            'num_attention_heads': 3,
            'use_mean_attention_aggr': True,
            'knn_graph_k': 48,
            'rbf_encoding_params': {'num_bins': 50, 'bin_min': 0.0, 'bin_max': 20.0},
        },

        'device': 'cuda:6',
        'dataset_path': '/scratch/bfry/torch_bioasmb_dataset' + ('/w7' if debug else ''),
        'clustering_output_prefix': 'torch_bioas_cluster30',
        'clustering_output_path': (output_path := '/scratch/bfry/bioasmb_dataset_sequence_clustering/'),
    }
    if params['debug']:
        print('Running in debug mode!')
    main(params)