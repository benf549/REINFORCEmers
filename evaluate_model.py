#!/usr/bin/env python3
import prody as pr
import torch
import torch.nn.functional as F
from collections import defaultdict
from typing import Dict, List, Optional

from utils.dataset import ClusteredDatasetSampler, UnclusteredProteinChainDataset, collate_sampler_data, BatchData
from utils.model import ReinforcemerRepacker
from utils.constants import dataset_atom_order, aa_idx_to_short, aa_idx_to_long, aa_short_to_idx
from torch.utils.data import DataLoader

from tqdm import tqdm

def create_prody_protein_from_coordinate_matrix(full_protein_coords, amino_acid_labels, bfactors: Optional[torch.Tensor] = None) -> pr.AtomGroup:

    # sanity check that the number of amino acid labels matches the number of coordinates
    assert(full_protein_coords.shape[0] == amino_acid_labels.shape[0])
    if bfactors is not None:
        assert(full_protein_coords.shape[0] == bfactors.shape[0])

    amino_acid_indices = amino_acid_labels.tolist()
    eval_mask = None
    if bfactors is not None:
        eval_mask = bfactors.tolist()

    all_coords = []
    prody_features = defaultdict(list)
    for resnum, (coord, aa_idx) in enumerate(zip(full_protein_coords, amino_acid_indices)):
        coord_mask = coord.isnan().any(dim=1)
        coord = coord[~coord_mask]
        atom_names = [x for idx, x in enumerate(dataset_atom_order[aa_idx_to_short[aa_idx]]) if not coord_mask[idx].item()]
        prody_features['atom_labels'].extend(atom_names)
        prody_features['resnames'].extend([aa_idx_to_long[aa_idx]] * len(atom_names))
        prody_features['resnums'].extend([resnum + 1] * len(atom_names))
        prody_features['chains'].extend(['A'] * len(atom_names))
        prody_features['occupancies'].extend([1.0] * len(atom_names))

        if bfactors is not None and eval_mask is not None:
            prody_features['bfactors'].extend([eval_mask[resnum]] * len(atom_names))

        all_coords.append(coord)
        if (~coord_mask).sum().item() != len(atom_names):
            print(resnum, aa_idx_to_long[aa_idx], coord, atom_names)
            raise NotImplementedError
    flattened_coords = torch.cat(all_coords, dim=0)

    assert(all([len(x) == len(flattened_coords) for x in prody_features.values()]))
    
    protein = pr.AtomGroup('Reinforcemers Repacked Protein')
    protein.setCoords(flattened_coords)
    protein.setNames(prody_features['atom_labels']) # type: ignore
    protein.setResnames(prody_features['resnames']) # type: ignore
    protein.setResnums(prody_features['resnums']) # type: ignore
    protein.setChids(prody_features['chains']) # type: ignore
    protein.setOccupancies(prody_features['occupancies'])# type: ignore
    if bfactors is not None and eval_mask is not None:
        protein.setBetas(prody_features['bfactors']) # type: ignore

    return protein

def batch_to_pdb_list(batch: BatchData, supervised_model: ReinforcemerRepacker, reinforced_model: ReinforcemerRepacker) -> Dict[str, List[pr.AtomGroup]]:
    # Set models to eval mode.
    supervised_model.eval()
    reinforced_model.eval()

    _, supervised_chi_samples = supervised_model(batch)
    _, reinforced_chi_samples = reinforced_model(batch)

    output = {'ground_truth': [], 'supervised_repack': [], 'reinforced_repack': []}
    # Iterate through batch indices.
    for idx, batch_index in enumerate(sorted(list(set(batch.batch_indices.numpy())))):
        curr_batch_mask = batch.batch_indices == batch_index

        curr_ground_truth_chi_angles = batch.chi_angles[curr_batch_mask]
        curr_backbone_coords = batch.backbone_coords[curr_batch_mask]
        curr_sequence_indices = batch.sequence_indices[curr_batch_mask]
        curr_supervised_sampled_chis = supervised_chi_samples[curr_batch_mask]
        curr_reinforced_sampled_chis = reinforced_chi_samples[curr_batch_mask]
        curr_eval_mask = ~batch.extra_atom_contact_mask[curr_batch_mask]

        # Mask out non-existant chi angles.
        native_chi_mask = curr_ground_truth_chi_angles.isnan()
        curr_supervised_sampled_chis[native_chi_mask] = torch.nan
        curr_reinforced_sampled_chis[native_chi_mask] = torch.nan

        fa_model = supervised_model.rotamer_builder.build_rotamers(curr_backbone_coords, curr_ground_truth_chi_angles, curr_sequence_indices)
        prody_protein = create_prody_protein_from_coordinate_matrix(fa_model, curr_sequence_indices, curr_eval_mask.float())
        output['ground_truth'].append(prody_protein)

        fa_model = supervised_model.rotamer_builder.build_rotamers(curr_backbone_coords, curr_supervised_sampled_chis, curr_sequence_indices)
        prody_protein = create_prody_protein_from_coordinate_matrix(fa_model, curr_sequence_indices, curr_eval_mask.float())
        output['supervised_repack'].append(prody_protein)

        fa_model = reinforced_model.rotamer_builder.build_rotamers(curr_backbone_coords, curr_reinforced_sampled_chis, curr_sequence_indices)
        prody_protein = create_prody_protein_from_coordinate_matrix(fa_model, curr_sequence_indices, curr_eval_mask.float())
        output['reinforced_repack'].append(prody_protein)

    return output


def main(params):
    # Load supervised learning weights.
    device = torch.device(params['device'])
    model_supervised = ReinforcemerRepacker(**params['model_params']).to(device)
    supervised_model_weights = torch.load(params['supervised_weights_input_prefix'], map_location=device)
    model_supervised.load_state_dict(supervised_model_weights)

    # Load reinforcement learning weights.
    model_reinforced = ReinforcemerRepacker(**params['model_params']).to(device)
    reinforced_model_weights = torch.load(params['reinforce_weights_input_prefix'], map_location=device)
    model_reinforced.load_state_dict(reinforced_model_weights)
    
    # Load test data.
    protein_dataset = UnclusteredProteinChainDataset(params)
    test_sampler = ClusteredDatasetSampler(protein_dataset, params, is_test_dataset_sampler=True)
    test_dataloader = DataLoader(protein_dataset, batch_sampler=test_sampler, collate_fn=collate_sampler_data, num_workers=params['num_workers'], persistent_workers=True)

    all_ground_truth_proteins = []
    all_supervised_repacked_proteins = []
    all_rl_repacked_proteins = []

    batch: BatchData
    for batch in tqdm(test_dataloader, total=len(test_dataloader)):
        batch.to_device(device)
        batch.construct_graph(0.0)

        atom_group_dict = batch_to_pdb_list(batch, model_supervised, model_reinforced)

        all_ground_truth_proteins.extend(atom_group_dict['ground_truth'])
        all_supervised_repacked_proteins.extend(atom_group_dict['supervised_repack'])
        all_rl_repacked_proteins.extend(atom_group_dict['reinforced_repack'])

    for idx in range(len(all_ground_truth_proteins)):
        pr.writePDB(f'./jerry_test_proteins/ground_truth/ground_truth_{idx:05}.pdb', all_ground_truth_proteins[idx])
        pr.writePDB(f'./jerry_test_proteins/supervised/supervised_{idx:05}.pdb', all_supervised_repacked_proteins[idx])
        pr.writePDB(f'./jerry_test_proteins/reinforced/reinforce_{idx:05}.pdb', all_rl_repacked_proteins[idx])
        

if __name__ == "__main__":
    torch.set_num_threads(10)
    params = {
        'debug': (debug := False),
        'supervised_weights_input_prefix': './model_weights/supervised_model_weights_teacher_forced_track_chi_acc_1_99.pt',
        'reinforce_weights_input_prefix': './model_weights/reinforce_finetuned_from_best_backprop_reward_amp_20.pt',
        'num_workers': 2,
        'num_epochs': 100,
        'batch_size': 10_000,
        'sample_randomly': False,
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
            'use_dense_chi_layer': False,
            'knn_graph_k': 24,
            'rbf_encoding_params': {'num_bins': 50, 'bin_min': 0.0, 'bin_max': 20.0},
        },
        'device': 'cpu',
        'dataset_path': '/scratch/bfry/torch_bioasmb_dataset' + ('/w7' if debug else ''),
        'clustering_output_prefix': 'torch_bioas_cluster30',
        'clustering_output_path': (output_path := '/scratch/bfry/bioasmb_dataset_sequence_clustering/'),
    }
    main(params)
