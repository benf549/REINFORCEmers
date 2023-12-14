import torch
import torch.nn.functional as F
from .constants import rotamer_alignment_tensor, leftover_atoms_tensor, ideal_bond_lengths_tensor, ideal_angles_tensor, ideal_aa_coords, aa_to_chi_angle_atom_index, aa_long2short, aa_to_chi_angle_mask, aa_name2aa_idx, dataset_atom_order, aa_long2short, aa_short2long, ATOM_IDENTITY_ENUM, DISULFIDE_S_CLASH_DIST, OTHER_ATOM_CLASH_DIST, METAL_CLASH_DIST, clash_matrix, aa_idx2aa_name, HARD_CLASH_TOLERANCE, CHI_BIN_MIN, CHI_BIN_MAX, amino_acid_to_atom_identity_matrix

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

