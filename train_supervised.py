#!/usr/bin/env python3

import os
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dataset import ClusteredDatasetSampler, UnclusteredProteinChainDataset, collate_sampler_data
from utils.build_rotamers import RotamerBuilder
from utils.model import ReinforcemerRepacker
from torch.utils.data import DataLoader 
from collections import defaultdict

import wandb

def main(params):
    # Load dataset, sample clusters to create batches of approximately batch_size, and load batches with dataloader.
    protein_dataset = UnclusteredProteinChainDataset(params)
    sampler = ClusteredDatasetSampler(protein_dataset, params)
    dataloader = DataLoader(protein_dataset, batch_sampler=sampler, collate_fn=collate_sampler_data, num_workers=params['num_workers'], persistent_workers=True)

    # Initialize device, model, and optimmizer for gradient descent.
    device = torch.device(params['device'])
    model = ReinforcemerRepacker(**params['model_params']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), params['learning_rate'])

    # Training loop.
    epoch_num = -1
    for epoch_num in range(params['num_epochs']):
        epoch_data = defaultdict(float)
        epoch_data['epoch'] = epoch_num
        for batch in tqdm(dataloader, total=len(dataloader), leave=False, desc=f'Training Epoch {epoch_num}'):
            # Initialize model for training.
            model.train()

            # Zero previous gradients
            optimizer.zero_grad()

            # Move batch to device
            batch.to_device(device)

            # Compute training mask: residues that are not in contact with extra atoms and being trained on.
            train_residue_mask = (~batch.extra_atom_contact_mask) & (~batch.chain_mask)
            
            # Compute mask of chi angles that are relevant the given amino acid.
            chi_mask = ~batch.chi_angles.isnan() 
            chi_mask = chi_mask[train_residue_mask]

            # Handle no valid chi angles in this batch.
            num_valid_residues = train_residue_mask.sum().item()
            if num_valid_residues == 0:
                continue

            # Compute supervised learning loss over residues that are not in contact with extra atoms, step loss and optimize.
            predicted_chi_angle_logits = model(batch)
            ground_truth_chi_angles = model.rotamer_builder.compute_binned_degree_basis_function(batch.chi_angles).nan_to_num()
            predicted_chi_angle_logits = predicted_chi_angle_logits[train_residue_mask]
            ground_truth_chi_angles = ground_truth_chi_angles[train_residue_mask]
            loss = F.cross_entropy(predicted_chi_angle_logits[chi_mask], ground_truth_chi_angles[chi_mask])
            # Debug NaN loss.
            if loss.isnan():
                import IPython; IPython.embed()
            loss.backward()
            optimizer.step()

            # Log debug information.
            epoch_data['loss'] += loss.item()
            epoch_data['num_samples'] += chi_mask.any(dim=1).sum() # Track number of residues that have at least one valid chi angle.

        # Average loss over all samples.
        epoch_data['loss'] /= max(epoch_data['num_samples'], 1)

        # Log metadata to wandb.
        if not params['debug']:
            wandb.log(dict(epoch_data))
        out = ""
        for key, value in epoch_data.items():
            out += f"{key}: {value} "
        print(out)

        # Save the model weights.
        if epoch_num % 5 == 0 and not params['debug']:
            torch.save(model.state_dict(), f"supervised_model_weights_{epoch_num}.pt")

    # Save the model weights.
    if not params['debug']:
        torch.save(model.state_dict(), f"supervised_model_weights_{epoch_num}.pt")


if __name__ == "__main__":
    params = {
        'debug': (debug := False),
        'num_workers': 2,
        'num_epochs': 100,
        'batch_size': 10_000,
        'learning_rate': 1e-4,
        'sample_randomly': True,
        'model_params': {
            'dropout': 0.1,
            'chi_angle_rbf_bin_width': 5,
            'node_embedding_dim': 128,
            'edge_embedding_dim': 128,
            'num_encoder_layers': 3,
            'num_attention_heads': 3,
            'use_mean_attention_aggr': True,
            'knn_graph_k': 24,
            'rbf_encoding_params': {'num_bins': 50, 'bin_min': 0.0, 'bin_max': 20.0},
        },
        'device': 'cuda:7',
        'dataset_path': '/scratch/bfry/torch_bioasmb_dataset' + ('/w7' if debug else ''),
        'clustering_output_prefix': 'torch_bioas_cluster30',
        'clustering_output_path': (output_path := '/scratch/bfry/bioasmb_dataset_sequence_clustering/'),
    }
    if params['debug']:
        print('Running in debug mode!')
    else:
        wandb.init(project='reinforcemers', entity='benf549', config=params)
    main(params)