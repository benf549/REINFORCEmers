#!/usr/bin/env python3

import os
import wandb
import subprocess
import prody as pr
from tqdm import tqdm
from typing import Optional
import multiprocessing as mp
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.build_rotamers import compute_chi_angle_accuracies
from utils.dataset import ClusteredDatasetSampler, UnclusteredProteinChainDataset, collate_sampler_data, BatchData
from utils.model import ReinforcemerRepacker
from utils.compute_reward import compute_pairwise_clash_penalties
from evaluate_model import create_prody_protein_from_coordinate_matrix


def compute_reinforce_loss(log_prob_action, reward):
    """
    Define the simplest REINFORCE loss we can optimize with gradient descent.
    """
    return -1 * log_prob_action * reward


@torch.no_grad()
def batch_to_pdb_list(batch: BatchData, reinforced_chi_samples: torch.Tensor, reinforced_model: ReinforcemerRepacker) -> list:
    # Set models to eval mode.
    reinforced_model.eval()

    # Build 3D coordinates for all residues in the protein with rotatable hydrogens.
    fa_model = reinforced_model.rotamer_builder.build_rotamers(batch.backbone_coords, reinforced_chi_samples, batch.sequence_indices)

    reinforced_chi_samples = reinforced_chi_samples.clone().cpu()
    fa_model = fa_model.detach().cpu()
    batch.to_device(torch.device('cpu'))

    output = []
    # Iterate through batch indices.
    for idx, batch_index in enumerate(sorted(list(set(batch.batch_indices.cpu().numpy())))):
        curr_batch_mask = batch.batch_indices == batch_index

        curr_sequence_indices = batch.sequence_indices[curr_batch_mask]
        curr_eval_mask = ~batch.extra_atom_contact_mask[curr_batch_mask]
        curr_fa_coords = fa_model[curr_batch_mask]

        prody_protein = create_prody_protein_from_coordinate_matrix(curr_fa_coords, curr_sequence_indices, curr_eval_mask.float())
        output.append(prody_protein)

    return output


def parse_probe_line(line):
    """
    Stolen from Combs2/dataset/probe.py
    Parses the probe output schema for relevant features.
    """
    spl = line.split(':')[1:]
    interaction = spl[1]
    chain1 = spl[2][:2].strip()
    resnum1 = int(spl[2][2:6])
    resname1 = spl[2][6:10].strip()
    name1 = spl[2][10:15].strip()
    atomtype1 = spl[12]
    chain2 = spl[3][:2].strip()
    resnum2 = int(spl[3][2:6])
    resname2 = spl[3][6:10].strip()
    name2 = spl[3][10:15].strip()
    atomtype2 = spl[13]
    return interaction, chain1, resnum1, resname1, name1, atomtype1, chain2, resnum2, resname2, name2, atomtype2


def compute_probe_metadata(protein_path, num_residues, device, include_wc=False):
    temp_path = protein_path.replace('.pdb', '_reduce.pdb')

    # Add hydrogens to protein.
    reduce_command = f"reduce -NOADJust {protein_path} > {temp_path}"
    subprocess.run(reduce_command, shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

    # Compute probe metadata.
    probe_command = f"probe -U -SEGID -CON -NOFACE -Explicit -MC -WEAKH -DE32 -4 -SE 'ALL' {temp_path}"
    output = subprocess.check_output(probe_command, shell=True, stderr=subprocess.DEVNULL).decode('utf-8')


    # there can be multiple interactions per residue so we need to store them in a set
    probe_dict_interactions = defaultdict(set)
    output_tensor = torch.zeros((num_residues,), dtype=torch.float16, device=device)
    for line in output.strip().split('\n'):
        line_data = parse_probe_line(line)
        residue = line_data[1:4]
        interaction = line_data[0]
        probe_dict_interactions[residue].add(interaction)

    # Process probe metadata for hydrogen bonds and clashes.
    for residue, interactions in probe_dict_interactions.items():
        # Hierarchically assigns value to each interaction type.
        if 'bo' in interactions:
            # Clash interaction type should be penalized
            interaction_value = -1.0
        elif 'hb' in interactions:
            # Regular hydrogen bond type should be rewarded
            interaction_value = 1.0
        elif 'wh' in interactions:
            # Weak hydrogen bond type should be rewarded
            interaction_value = 0.5
        elif 'so' in interactions:
            # Strong overlap (redundant with bo)
            interaction_value = 0
        elif 'cc' in interactions:
            # Close Contact (redundant with bo)
            interaction_value = 0
        elif include_wc and 'wc' in interactions:
            # Water mediated contact irrelevant unless we place waters somehow.
            interaction_value = 0
        else:
            continue
        output_tensor[residue[1] - 1] = interaction_value
    return output_tensor


def compute_probe_reward(protein_list, device):
    """
    This is super inefficient but whatever lol.
    """
    output_list = []
    temp_path = '/scratch/bfry/temp2/'
    for protein in protein_list:
        # Write protein to disk.
        protein_path = os.path.join(temp_path, 'test_protein.pdb')
        pr.writePDB(protein_path, protein)

        # Compute probe metadata from pdb file.
        output_list.append(compute_probe_metadata(protein_path, protein.numResidues(), device))
    return torch.cat(output_list, dim=0)


def process_epoch(model: ReinforcemerRepacker, optimizer: Optional[torch.optim.Adam], dataloader: DataLoader, epoch_num: int, params: dict):

    is_train_epoch = optimizer is not None

    # Loop over batches in dataloader.
    epoch_list_data = defaultdict(list)
    epoch_data = defaultdict(float)
    batch: BatchData
    for batch in tqdm(dataloader, total=len(dataloader), leave=False, desc=f'{"Training" if is_train_epoch else "Testing"} Epoch {epoch_num}'):

        # Zero previous gradients if in training epoch.
        if is_train_epoch:
            optimizer.zero_grad()
        
        # Move batch to device, build graph and perturb by noise if a train epoch to reduce overfitting.
        batch.to_device(model.device)
        chi_angles_original = batch.chi_angles.clone()
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

        # REINFORCE loss over residues that are not in contact with extra atoms, step loss and optimize.
        chi_logits, sampled_chi_angles = model(batch)

        # Extract just the log probability of the sampled 'actions' (chi angles).
        log_probs = F.log_softmax(chi_logits, dim=-1)
        sampled_chi_indices = model.rotamer_builder.compute_binned_degree_basis_function(sampled_chi_angles).argmax(dim=-1)
        log_prob_action = log_probs.gather(2, sampled_chi_indices.unsqueeze(-1)).squeeze(-1)
        log_prob_action = log_prob_action[valid_residue_mask]

        # Compute reward for each sampled chi angle.
        pdb_list = batch_to_pdb_list(batch, sampled_chi_angles, model)
        reward = compute_probe_reward(pdb_list, model.device)

        # Reward is negative or positive penalties.
        masked_reward = reward[valid_residue_mask]
        masked_reward_exp = masked_reward.unsqueeze(-1).expand(-1, 4)[chi_mask]

        # Compute REINFORCE loss.
        loss = compute_reinforce_loss(log_prob_action[chi_mask], masked_reward_exp).sum()
        loss = loss / max(chi_mask.any(dim=1).sum().item(), 1)

        # Step for gradient descent if optimizer is provided.
        if is_train_epoch:
            loss.backward()
            optimizer.step()

        # Compute chi angle recovery/accuracies.
        chi_accuracy = compute_chi_angle_accuracies(sampled_chi_angles[valid_residue_mask], chi_angles_original[valid_residue_mask], model.rotamer_builder)
        for chi_acc, acc_value in chi_accuracy.items():
            epoch_list_data[chi_acc].append(acc_value)

        # Log debug information.
        epoch_data['loss'] += loss.item()
        epoch_data['reward'] += masked_reward_exp.sum().item()
        epoch_data['num_samples'] += chi_mask.any(dim=1).sum().item() # Track number of residues that have at least one valid chi angle.

    # Average loss over all samples.
    epoch_data['loss'] /= max(epoch_data['num_samples'], 1)
    epoch_data['reward'] /= max(epoch_data['num_samples'], 1)
    epoch_list_data = {x: sum(y) / max(len(y), 1) for x,y in epoch_list_data.items()}
    return {**epoch_data, **epoch_list_data}


def train_epoch(model: ReinforcemerRepacker, optimizer: torch.optim.Adam, dataloader: DataLoader, epoch_num: int, params: dict):
    model.train()
    epoch_data = process_epoch(model, optimizer, dataloader, epoch_num, params)
    return {'train_' + x: y for x,y in epoch_data.items()}


@torch.no_grad()
def test_epoch(model: ReinforcemerRepacker, dataloader: DataLoader, epoch_num: int, params: dict) -> dict:
    """
    Evaluate model on test set. model.eval() enables autoregressive sampling.
    """
    model.eval()
    epoch_data = process_epoch(model, None, dataloader, epoch_num, params)
    return {'test_' + x: y for x,y in epoch_data.items()}


def main(params: dict) -> None:
    # Initialize device, model, and optimmizer for gradient descent.
    device = torch.device(params['device'])
    model = ReinforcemerRepacker(**params['model_params']).to(device)

    if params['use_supervised_learning_weights']:
        print("Loading model weights from supervised learning model:", params['model_input_weights_path'])
        model.load_state_dict(torch.load(params['model_input_weights_path'], map_location=device))

    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])

    # Load dataset
    protein_dataset = UnclusteredProteinChainDataset(params)

    # sample train clusters to create batches of approximately batch_size, and load batches with dataloader.
    train_sampler = ClusteredDatasetSampler(protein_dataset, params, is_test_dataset_sampler=False)
    train_dataloader = DataLoader(protein_dataset, batch_sampler=train_sampler, collate_fn=collate_sampler_data, num_workers=params['num_workers'], persistent_workers=True)

    # sample test clusters.
    test_sampler = ClusteredDatasetSampler(protein_dataset, params, is_test_dataset_sampler=True)
    test_dataloader = DataLoader(protein_dataset, batch_sampler=test_sampler, collate_fn=collate_sampler_data, num_workers=params['num_workers'], persistent_workers=True)

    epoch_num = -1
    for epoch_num in range(params['num_epochs']):
        # Train the model for an epoch.
        train_epoch_data = train_epoch(model, optimizer, train_dataloader, epoch_num, params)

        # Test model every few epochs.
        test_epoch_data = {}
        if epoch_num % 5 == 0:
            test_epoch_data = test_epoch(model, test_dataloader, epoch_num, params)
            if not params['debug']:
                torch.save(model.state_dict(), f"{params['weights_output_prefix']}_{epoch_num}.pt")

        # Combine train and test metadata.
        epoch_data = {**train_epoch_data, **test_epoch_data, 'epoch': epoch_num}

        # Log metadata to wandb.
        if params['use_wandb']:
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
        'debug': (debug := True),
        'use_wandb': True and not debug,
        'use_supervised_learning_weights': (use_pretrained_model := True),
        'model_input_weights_path': ('./model_weights/supervised_model_weights_teacher_forced_track_chi_acc_1_80.pt' if use_pretrained_model else None),
        'weights_output_prefix': './model_weights/debug_probe_finetuned',
        'num_workers': 2,
        'num_epochs': 100,
        'batch_size': 10_000,
        'max_protein_size': 500,
        'max_clash_penalty': 1,
        'learning_rate': 1e-8,
        'training_noise': 0.05,
        'sample_randomly': True,
        'train_splits_path': ('./files/train_splits_debug.pt' if debug else './files/train_splits.pt'),
        'test_splits_path': ('./files/test_splits_debug.pt' if debug else './files/test_splits.pt'),
        'model_params': {
            'disable_teacher_forcing': True,
            'dropout': 0.1,
            'chi_angle_rbf_bin_width': 5,
            'node_embedding_dim': 128,
            'edge_embedding_dim': 128,
            'num_encoder_layers': 3,
            'num_attention_heads': 3,
            'use_mean_attention_aggr': True,
            'use_dense_chi_layer': False,
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
        if params['use_wandb']:
            wandb.init(project='probe_reinforcemers', entity='benf549', config=params)
    main(params)