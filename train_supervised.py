#!/usr/bin/env python3

import os
import wandb
import pandas as pd

from tqdm import tqdm
from typing import Optional

import torch
import torch.nn.functional as F

from utils.build_rotamers import compute_num_chi_correct
from utils.dataset import ClusteredDatasetSampler, UnclusteredProteinChainDataset, collate_sampler_data
from utils.model import ReinforcemerRepacker
from torch.utils.data import DataLoader 
from collections import defaultdict



def process_epoch(model: ReinforcemerRepacker, optimizer: Optional[torch.optim.Adam], dataloader: DataLoader, epoch_num: int) -> dict:
    """
    Handle a single epoch of training or testing.
        Assumes model is in train mode if optimizer is provided, and eval mode otherwise.
    """
    is_train_epoch = optimizer is not None

    # Loop over batches in dataloader.
    epoch_list_data = defaultdict(list)
    epoch_data = defaultdict(float)
    for batch in tqdm(dataloader, total=len(dataloader), leave=False, desc=f'{"Training" if is_train_epoch else "Testing"} Epoch {epoch_num}'):

        # Zero previous gradients if in training epoch.
        if is_train_epoch:
            optimizer.zero_grad()

        # Move batch to device, build graph and perturb by noise if a train epoch to reduce overfitting.
        batch.to_device(model.device)
        batch.construct_graph(params['training_noise'] if is_train_epoch else 0.0)

        # Compute training mask: residues that are not in contact with extra atoms and being trained on.
        valid_residue_mask = (~batch.extra_atom_contact_mask) & (~batch.chain_mask)
        
        # Compute mask of chi angles that are relevant the given amino acid.
        chi_mask = ~batch.chi_angles.isnan() 
        chi_mask = chi_mask[valid_residue_mask]

        # Handle no valid chi angles in this batch.
        num_valid_residues = valid_residue_mask.sum().item()
        if num_valid_residues == 0:
            continue

        # Compute supervised learning loss over residues that are not in contact with extra atoms, step loss and optimize.
        chi_logits, sampled_chi_angles = model(batch)
        ground_truth_chi_angles = model.rotamer_builder.compute_binned_degree_basis_function(batch.chi_angles).nan_to_num()

        chi_logits = chi_logits[valid_residue_mask]
        ground_truth_chi_angles = ground_truth_chi_angles[valid_residue_mask]

        # Compute supervised learning loss.
        loss = F.cross_entropy(chi_logits[chi_mask], ground_truth_chi_angles[chi_mask], reduction='sum')
        loss /= max(chi_mask.any(dim=1).sum().item(), 1)

        # Step for gradient descent if optimizer is provided.
        if is_train_epoch:
            loss.backward()
            optimizer.step()

        chi_accuracy = compute_num_chi_correct(sampled_chi_angles, batch.chi_angles, model.rotamer_builder)
        for chi_acc, acc_value in chi_accuracy.items():
            epoch_list_data[chi_acc].append(acc_value)
        # Log debug information.
        epoch_data['loss'] += loss.item()
        epoch_data['num_samples'] += chi_mask.any(dim=1).sum() # Track number of residues that have at least one valid chi angle.

    # Average loss over all samples.
    epoch_data['loss'] /= max(epoch_data['num_samples'], 1)
    epoch_list_data = {x: sum(y) / max(len(y), 1) for x,y in epoch_list_data.items()}
    return {**epoch_data, **epoch_list_data}


def train_epoch(model: ReinforcemerRepacker, optimizer: torch.optim.Adam, dataloader: DataLoader, epoch_num: int) -> dict:
    """
    Train model on train set.
    """
    model.train()
    epoch_data = process_epoch(model, optimizer, dataloader, epoch_num)
    return {'train_' + x: y for x,y in epoch_data.items()}


@torch.no_grad()
def test_epoch(model: ReinforcemerRepacker, dataloader: DataLoader, epoch_num: int) -> dict:
    """
    Evaluate model on test set.
    """
    model.eval()
    epoch_data = process_epoch(model, None, dataloader, epoch_num)
    return {'test_' + x: y for x,y in epoch_data.items()}


def main(params: dict) -> None:
    # Initialize device, model, and optimmizer for gradient descent.
    device = torch.device(params['device'])
    model = ReinforcemerRepacker(**params['model_params']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), params['learning_rate'])

    # Load dataset
    protein_dataset = UnclusteredProteinChainDataset(params)

    # sample train clusters to create batches of approximately batch_size, and load batches with dataloader.
    train_sampler = ClusteredDatasetSampler(protein_dataset, params, is_test_dataset_sampler=False)
    train_dataloader = DataLoader(protein_dataset, batch_sampler=train_sampler, collate_fn=collate_sampler_data, num_workers=params['num_workers'], persistent_workers=True)

    test_sampler = ClusteredDatasetSampler(protein_dataset, params, is_test_dataset_sampler=True)
    test_dataloader = DataLoader(protein_dataset, batch_sampler=test_sampler, collate_fn=collate_sampler_data, num_workers=params['num_workers'], persistent_workers=True)

    # Training loop.
    epoch_num = -1
    for epoch_num in range(params['num_epochs']):

        # Train the model for an epoch.
        train_epoch_data = train_epoch(model, optimizer, train_dataloader, epoch_num)

        # Test model every few epochs.
        test_epoch_data = {}
        if epoch_num % 5 == 0:
            test_epoch_data = test_epoch(model, test_dataloader, epoch_num)
            if not params['debug']:
                torch.save(model.state_dict(), f"{params['weights_output_prefix']}_{epoch_num}.pt")

        # Combine train and test metadata.
        epoch_data = {**train_epoch_data, **test_epoch_data, 'epoch': epoch_num}

        # Log metadata to wandb.
        if not params['debug']:
            wandb.log(dict(epoch_data))
        
        # Print training data to console.
        out = []
        for key, value in epoch_data.items():
            out.append(f"{key}: {value:0.6f}")
        print(', '.join(out))

    # Save the model weights.
    if not params['debug']:
        torch.save(model.state_dict(), f"{params['weights_output_prefix']}_{epoch_num}.pt")


if __name__ == "__main__":
    params = {
        'debug': (debug := False),
        'weights_output_prefix': './model_weights/supervised_model_weights_teacher_forced_track_chi_acc',
        'num_workers': 2,
        'num_epochs': 100,
        'batch_size': 10_000,
        'learning_rate': 1e-4,
        'training_noise': 0.05,
        'sample_randomly': True,
        'train_splits_path': ('./files/train_splits_debug.pt' if debug else './files/train_splits.pt'),
        'test_splits_path': ('./files/test_splits_debug.pt' if debug else './files/test_splits.pt'),
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
        'device': 'cuda:0',
        'dataset_path': '/scratch/bfry/torch_bioasmb_dataset' + ('/w7' if debug else ''),
        'clustering_output_prefix': 'torch_bioas_cluster30',
        'clustering_output_path': (output_path := '/scratch/bfry/bioasmb_dataset_sequence_clustering/'),
    }
    if params['debug']:
        print('Running in debug mode!')
    else:
        wandb.init(project='reinforcemers', entity='benf549', config=params)
    main(params)