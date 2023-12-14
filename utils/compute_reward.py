import torch
import torch.nn.functional as F
import numpy as np
from .constants import rotamer_alignment_tensor, leftover_atoms_tensor, ideal_bond_lengths_tensor, ideal_angles_tensor, ideal_aa_coords, aa_to_chi_angle_atom_index, aa_long2short, aa_to_chi_angle_mask, aa_name2aa_idx, dataset_atom_order, aa_long2short, aa_short2long, ATOM_IDENTITY_ENUM, DISULFIDE_S_CLASH_DIST, OTHER_ATOM_CLASH_DIST, METAL_CLASH_DIST, clash_matrix, aa_idx2aa_name, HARD_CLASH_TOLERANCE, CHI_BIN_MIN, CHI_BIN_MAX, amino_acid_to_atom_identity_matrix, hbond_candidate_set, hbond_mask_dict, hbond_element_dict ,HBOND_CAPABLE_ELEMENTS, HBOND_MAX_DISTANCE, HBOND_MAX_DISTANCE, ON_TO_S_HBOND_MAX_DISTANCE, S_TO_S_HBOND_MAX_DISTANCE, hbond_candidate_indices

SOFT_CLASH_THRESHOLD = 2.5
    

def clash_loss_penalty(distances, residue_types_source, residue_types_sink, atom_types_source, atom_types_sink, clash_matrix_dev, max_penalty):
    cysteine_index = aa_name2aa_idx['CYS']

    atom_types_source = atom_types_source.unsqueeze(-1).expand(-1, 15, 15)
    atom_types_sink = atom_types_sink.unsqueeze(-1).expand(-1, 15, 15)

    residue_types_source = residue_types_source.unsqueeze(-1).unsqueeze(-1).expand(-1, 15, 15)
    residue_types_sink = residue_types_sink.unsqueeze(-1).unsqueeze(1).expand(-1, 15, 15)

    clash_distances = clash_matrix_dev[atom_types_source, atom_types_sink]

    atom_pair_mask = (
        ((residue_types_source == cysteine_index) & (residue_types_sink == cysteine_index)) &
        ((atom_types_source == ATOM_IDENTITY_ENUM.index('S')) & (atom_types_sink == ATOM_IDENTITY_ENUM.index('S')))
    )
    clash_distances[atom_pair_mask] = DISULFIDE_S_CLASH_DIST - HARD_CLASH_TOLERANCE

    atom_pair_mask = (
        ((residue_types_source == aa_name2aa_idx['CYS']) & (residue_types_sink == aa_name2aa_idx['CYS'])) & (
            ((atom_types_source == ATOM_IDENTITY_ENUM.index('C')) & (atom_types_sink == ATOM_IDENTITY_ENUM.index('S'))) | 
            ((atom_types_sink == ATOM_IDENTITY_ENUM.index('S')) & (atom_types_sink == ATOM_IDENTITY_ENUM.index('C')))
        )
    )
    clash_distances[atom_pair_mask] = OTHER_ATOM_CLASH_DIST - HARD_CLASH_TOLERANCE

    #TODO: try inverse square penalty
    penalties = max_penalty * (1 - (distances / clash_distances))
    penalties.clamp_(min=0.0)

    # Mask tracking which atom-atom distances are backbone-backbone. the rest are backbone-sidechain and sidechain-sidechain.
    not_backbone_backbone_mask = torch.ones(15, 15, device=distances.device, dtype=torch.bool)
    not_backbone_backbone_mask[:5, :5] = False

    # Convert NaN values to 0.0, resulting (E, 200) tensor of pairwise distances between all relevant atoms in the two residues.
    penalties = penalties[:, not_backbone_backbone_mask].nan_to_num()

    return penalties


def compute_pairwise_clash_penalties(placed_aligned_rotamers, bb_bb_eidx, bb_label_indices, penalty_max=100, return_residue_indices=False):
    """
    Identifies residues involved in sidechain-sidechain and sidechain-backbone clashes for a batch of proteins with eachother.

    Uses the graph structure stored in bb_bb_eidx to only compute the distances between residues that are connected in the KNN graph.
    Given the full atomistic coordinates of the protein as input in the (N, 15, 3) tensor placed_aligned_rotamers, computes the distances of all atoms in each residues to all atoms in their K nearest neighbors.
    Applies clash_loss_penalty function to every pairwise distance to compute the penalty for each pair of residues. 
    Takes the smallest distance implicated in the clash as the penalty for the pair of residues and sums this over all clashes for total penalty.
    """
    aa_to_atom_identity = amino_acid_to_atom_identity_matrix.to(placed_aligned_rotamers.device)

    # Mask tracking self-edges in the KNN graph.  Use to only look at edges between different residues.
    residue_indices = torch.arange(placed_aligned_rotamers.shape[0], device=placed_aligned_rotamers.device)
    same_residue_mask = residue_indices[bb_bb_eidx[0]] == residue_indices[bb_bb_eidx[1]]
    neighbor_eidces = bb_bb_eidx[:, ~same_residue_mask]

    # For each pair of interactions in neighbor_eidces, compute the distance between all atoms in the two residues.
    source_coords = placed_aligned_rotamers[neighbor_eidces[0]]
    sink_coords = placed_aligned_rotamers[neighbor_eidces[1]]

    # Get indexed representations of the atom types involved in each pairwise distance that will be computed to threshold what defines a clash.
    residue_types_source = bb_label_indices[neighbor_eidces[0]]
    residue_types_sink = bb_label_indices[neighbor_eidces[1]]
    atom_types_source = aa_to_atom_identity[residue_types_source]
    atom_types_sink = aa_to_atom_identity[residue_types_sink]

    clash_matrix_dev = clash_matrix.to(placed_aligned_rotamers.device)

    # (E, 15, 15) tensor of pairwise distances between all atoms in the two residues.
    clash_penalties = torch.cdist(source_coords, sink_coords)
    clash_penalties = clash_loss_penalty(clash_penalties, residue_types_source, residue_types_sink, atom_types_source, atom_types_sink, clash_matrix_dev, penalty_max)

    # Take the highest clash penalty computed for two atoms in each residue pair and sum over all residue pairs.
    clash_penalties = clash_penalties.amax(dim=1)


    if return_residue_indices:
        return clash_penalties, neighbor_eidces
    
    return clash_penalties

def compute_rotamer_clash_penalty(placed_aligned_rotamers, bb_bb_eidx, bb_label_indices, penalty_max=100):
    """
    Takes the smallest distance implicated in a clash as the penalty for the pair of residues and sums this over all clashes for total penalty.
    """

    # Compute the pairwise clash penalties for all pairs of residues.
    clash_penalties = compute_pairwise_clash_penalties(placed_aligned_rotamers, bb_bb_eidx, bb_label_indices, penalty_max)
    
    return clash_penalties.sum() / placed_aligned_rotamers.shape[0]


def pad_matrix_with_nan(data, target_size, last_dim=True):
    """
    Given some target size (target_size) for the last dimension of a torch tensor, 
    pads the existing tensor with np.NaN up to that size along that last dimension.
    """
    if last_dim:
        # Pad along last dimension.
        last_dim_size = data.shape[-1]
        if last_dim_size < target_size:
            return F.pad(data.float(), (0, target_size - data.shape[-1]), 'constant', np.nan)
        else:
            return data
    else:
        # Pad along first dimension.
        first_dim_size = data.shape[0]
        if first_dim_size < target_size:
            return F.pad(data.float(), (0, 0, 0, target_size - data.shape[0]), 'constant', np.nan)
        else:
            return data
        
def identify_sidechain_hydrogen_bonding_coordinates(hbond_capable_coords, hbond_candidate_mask, label_indices):
    """
    Iterate through the residues capable of forming sidechain-mediated hydrogen bonds and record the
    coordinates of the atoms that can form hydrogen bonds and element identities of those atoms.
    """
    padded_sc_coords = []
    atom_element_list = []
    for idx, aa_label_idx in enumerate(label_indices[hbond_candidate_mask].cpu().tolist()):
        residue_name = aa_idx2aa_name[aa_label_idx]
        padded_sc_coords.append(pad_matrix_with_nan(hbond_capable_coords[idx, hbond_mask_dict[residue_name]], 3, last_dim=False))
        sc_hbonding_atom_elements = hbond_element_dict[residue_name]

        for i in range(3 - len(sc_hbonding_atom_elements)):
            sc_hbonding_atom_elements.append(HBOND_CAPABLE_ELEMENTS.index('Padding'))
        atom_element_list.extend(sc_hbonding_atom_elements)
    return padded_sc_coords, torch.tensor(atom_element_list)

def compute_hbonding_adjacency_matrix(all_sc_distance_mtx, atom_element_list, num_putative_hbonding, device=torch.device('cpu')):
    """
    Defines a hydrogen bond (quick to compute, not necessarily geometrically valid) by pairwise atom distances.
    Implements different distance thresholds to define what an 'edge' is for hydrogen bonding which are selected 
    based on the pair of atom identities that the distance is computed for.

    Checks for hydrogen bonds between all pairs of Nitrogen, Oxygen, and Sulfur found in sidechains.
    """
    # Construct an 2 x N x N tensor tracking every pair of elements for a given adjacency matrix edge.
    pairwise_element_encoding = torch.stack([
        atom_element_list.unsqueeze(1).expand(all_sc_distance_mtx.shape), 
        atom_element_list.expand(all_sc_distance_mtx.shape)
    ]).to(device)

    # If a pair of nitrogen and oxygen 
    nitrogen_mask = (pairwise_element_encoding == HBOND_CAPABLE_ELEMENTS.index('N'))
    oxygen_mask = (pairwise_element_encoding == HBOND_CAPABLE_ELEMENTS.index('O'))
    sulfur_mask = (pairwise_element_encoding == HBOND_CAPABLE_ELEMENTS.index('S'))

    # Count number of oxygens and nitrogens in the pair at a given index.
    nitrogen_or_oxygen_mask = (nitrogen_mask | oxygen_mask).sum(dim=0)
    # True if both atoms are sulfur.
    sulfur_and_sulfur_mask = sulfur_mask.sum(dim=0) == 2
    # True if one index is nitrogen or oxygen and the other is sulfur.
    sulfur_and_NorO_mask = ((nitrogen_or_oxygen_mask == 1) & (sulfur_mask.sum(dim=0) > 0) & (~sulfur_and_sulfur_mask))
    
    A = ( (all_sc_distance_mtx < HBOND_MAX_DISTANCE) & (nitrogen_or_oxygen_mask == 2))
    B = ( (all_sc_distance_mtx < ON_TO_S_HBOND_MAX_DISTANCE) & sulfur_and_NorO_mask)
    C = ( (all_sc_distance_mtx < S_TO_S_HBOND_MAX_DISTANCE) & sulfur_and_sulfur_mask)
    adjacency_matrix = A + B + C
    adjacency_matrix = torch.stack(torch.split(adjacency_matrix, 3)).sum(dim=1)
    adjacency_matrix = torch.stack(torch.split(adjacency_matrix, 3, dim=1)).sum(dim=2) > 0
    adjacency_matrix = (1 - torch.eye(num_putative_hbonding, num_putative_hbonding, device=device)).bool() & adjacency_matrix

    return adjacency_matrix

def generate_hydrogen_bonding_graph(full_residue_coords, bb_labels, sampled_aa, device=torch.device('cpu')):
    """
    Attempts to search for sidechain-mediated hydrogen bonding networks around the ligand to penalize in model loss.

    Compute the graph of residues that are capable of hydrogen bonding to eachother.
    Returns None if sampled residue does not have a sidechain O/N residue.
    """
    # Check if the residue we sampled is capable of hydrogen bonding, if not there is no need to look for hydrogen bonding networks
    if not sampled_aa in hbond_candidate_set:
        return None, None
    
    label_indices = bb_labels.argmax(dim=1)
    hbond_candidate_mask = torch.isin(label_indices, hbond_candidate_indices.to(device))
    num_putative_hbonding = hbond_candidate_mask.sum().item()

    hbond_capable_coords = full_residue_coords[hbond_candidate_mask, 4:]
    padded_sc_coords, atom_element_list = identify_sidechain_hydrogen_bonding_coordinates(hbond_capable_coords, hbond_candidate_mask, label_indices)

    # Flatten to compute distances between all N/O atoms.
    padded_sc_coords = torch.cat(padded_sc_coords, dim=0).to(device)
    all_sc_distance_mtx = torch.cdist(padded_sc_coords, padded_sc_coords)

    # Convert distance matrix to an adjacency matrix with reduction to residue/residue interaction and self interactions removed.
    adjacency_matrix = compute_hbonding_adjacency_matrix(all_sc_distance_mtx, atom_element_list, num_putative_hbonding, device)

    return adjacency_matrix, hbond_candidate_mask