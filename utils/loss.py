import os

import torch

import subprocess
from utils.dataset import BatchData
from utils.model import ReinforcemerRepacker
from utils.constants import dataset_atom_order, aa_idx_to_short, aa_idx_to_long
from collections import defaultdict
import prody as pr
from typing import Optional


def compute_reinforce_loss(log_prob_action, reward):
    """
    Define the simplest REINFORCE loss we can optimize with gradient descent.
    """
    return -1 * log_prob_action * reward


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


def compute_probe_metadata(protein_path, num_residues, include_wc=False):
    temp_path = protein_path.replace('.pdb', '_reduce.pdb')

    # Add hydrogens to protein.
    reduce_command = f"reduce -NOADJust {protein_path} > {temp_path}"
    subprocess.run(reduce_command, shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

    # Compute probe metadata.
    probe_command = f"probe -U -SEGID -CON -NOFACE -Explicit -MC -WEAKH -DE32 -4 -SE 'ALL' {temp_path}"
    output = subprocess.check_output(probe_command, shell=True, stderr=subprocess.DEVNULL).decode('utf-8')

    # there can be multiple interactions per residue so we need to store them in a set
    probe_dict_interactions = defaultdict(set)
    output_tensor = torch.zeros((num_residues,), dtype=torch.float16).share_memory_()
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

    # Clean up temp paths.
    os.remove(protein_path)
    os.remove(temp_path)

    return output_tensor

def worker_compute_probe_reward(tup):
    idx, protein, temp_path = tup

    # Write protein to disk.
    output_path = os.path.join(temp_path, f'test_protein_{idx}.pdb')
    pr.writePDB(output_path, protein)

    # Compute probe metadata from pdb file.
    output_tensor = compute_probe_metadata(output_path, protein.numResidues())
    return idx, output_tensor

def compute_probe_reward(protein_list, worker_pool, device):

    output_list = [None] * len(protein_list)
    temp_path = '/scratch/bfry/temp2/'
    pool_data = [(idx, protein, temp_path) for idx, protein  in enumerate(protein_list)]
    for out_idx, out_tensor in worker_pool.imap(worker_compute_probe_reward, pool_data):
        output_list[out_idx] = out_tensor.to(device)

    return torch.cat(output_list, dim=0) #type: ignore


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
