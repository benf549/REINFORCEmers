"""
Implements a RotamerBuilder class that builds full atom models of proteins given chi angles.
Should be used with atom orderings as defined in constants.py.

Benjamin Fry (bfry@g.harvard.edu) 
12/6/2023
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, Optional
from utils.constants import MAX_NUM_RESIDUE_ATOMS, CHI_BIN_MIN, CHI_BIN_MAX, ideal_aa_coords, ideal_bond_lengths, ideal_bond_angles, aa_to_chi_angle_atom_index, aa_to_leftover_atoms, alignment_indices, aa_short_to_idx


class RotamerBuilder(nn.Module):
    """
    A class for building full atom models of proteins given chi angles.
    Also used for embedding chi angles in an angular RBF.
    Built into a torch module to put all of the helper tensors on the GPU if necessary.
    """
    def __init__(self, chi_angle_rbf_bin_width, **kwargs):
        super(RotamerBuilder, self).__init__()

        # Set the width of the bins for the chi angles.
        self.chi_angle_rbf_bin_width = chi_angle_rbf_bin_width

        # Pad ideal coords with NaNs to allow for indexing with MAX_NUM_RESIDUE_ATOMS indices.
        self.register_buffer('ideal_aa_coords', F.pad(ideal_aa_coords, (0, 0, 0, 1, 0, 0), 'constant', torch.nan))
        self.register_buffer('ideal_bond_lengths', ideal_bond_lengths)
        self.register_buffer('ideal_bond_angles', ideal_bond_angles)
        self.register_buffer('aa_to_chi_angle_atom_index', aa_to_chi_angle_atom_index)
        self.register_buffer('leftover_atom_indices', aa_to_leftover_atoms)
        self.register_buffer('alignment_indices', alignment_indices)
        self.register_buffer('backbone_frame_mask_1', torch.tensor([True, True, False, True, False]))
        self.register_buffer('backbone_frame_mask_2', torch.tensor([True, True, False, True, True]))

        # Compute the number of chi angle bins, the embedding dimensionality, and the index to degree bin tensor.
        self.num_chi_bins = int((180 * 2) / self.chi_angle_rbf_bin_width)
        self.chi_angle_embed_dim = self.num_chi_bins * 4
        self.register_buffer('index_to_degree_bin', torch.arange(CHI_BIN_MIN, CHI_BIN_MAX, self.chi_angle_rbf_bin_width).float())

    def compute_binned_degree_basis_function(self, degrees: torch.Tensor, std_dev: Optional[float] = None) -> torch.Tensor:
        """
        Given degrees tensor (N, 4) of degrees between -180 and 180, computes the density of that value in circularly symmetric bin space.
            Degree values can be NaN.

        Basically implements an RBF for degrees between [-180, 180) with a standard deviation of half the bin width in degrees by 
            default in modulus 360 degrees so -180 and 179 are 1 degree apart.

        Each bin is a gaussian centered at the bin center (increments of 5 degrees) with a standard deviation of 2.5 degrees by default.
        """

        # Default standard deviation to half the bin width in degrees.
        if std_dev is None:
            std_dev = self.chi_angle_rbf_bin_width / 2
        
        # Reshape the input to allow subtraction broadcasting with the bin centers.
        degree_input_shape = degrees.shape
        if degrees.dim() != 1:
            degrees = degrees.flatten().unsqueeze(-1)

        # Compute offset in degrees relative to the current angle in circular bin space.
        bin_degrees_exp = self.index_to_degree_bin.expand(degrees.shape[0], -1) # type: ignore
        circular_bin_distance = torch.minimum(torch.remainder(bin_degrees_exp - degrees, 360), torch.remainder(degrees - bin_degrees_exp, 360))

        # Encode offset in a gaussian with standard deviation of 5 degrees by default meaning the width of 1 bin is within +/- 1 of std_dev
        A = torch.exp((-1/2) * (circular_bin_distance / std_dev) ** 2)

        # Returns encoding normalized to sum to 1
        # NOTE: Looks like input to cross-entropy does not need to be normalized (revisit if using for supervised learning).
        out = A / A.sum(dim=1).unsqueeze(-1)

        return out.reshape(*degree_input_shape, -1)


    def compute_backbone_alignment_matrices(self, full_atom_coords):
        """
        Give a way for user to pass in full_atom_coord tensor and get the alignment tensors for aligning ideal coordinates to backbone.
        """
        raise NotImplementedError

    def place_tyr_hydrogens(self, tyr_coords: torch.Tensor, sequence_indices: torch.Tensor, chi_3_angles: torch.Tensor):
        """
        Places the 'HH' hydrogen atom of a tyrosine residue based on the given coordinates, sequence indices, and chi_3 angles.
        Should be called after the rest of the atoms have been placed since the ring atoms get rotated by alignment process.

        Args:
            tyr_coords (torch.Tensor): The coordinates of the tyrosine residue (M, 15, 3)
            sequence_indices (torch.Tensor): The sequence indices of the tyrosine residue (M,)
            chi_3_angles (torch.Tensor): The chi_3 angles of the tyrosine residue. (M,)

        Returns:
            torch.Tensor: The updated coordinates of the tyrosine residue with the hydrogen atoms placed. (M, 15, 3)
        """
        # Replaces tyr hydrogens after placing the rest of the atoms. Exactly the same as adjust_chi_rotatable_ideal_atom_placements.
        chi_number = 2
        curr_indices = self.aa_to_chi_angle_atom_index[sequence_indices] # type: ignore

        prev_coords = tyr_coords.gather(1, curr_indices[:, chi_number, :3].unsqueeze(-1).expand(-1, -1, 3))
        ideal_bond_lengths = self.ideal_bond_lengths[sequence_indices][:, chi_number].unsqueeze(-1) # type: ignore
        ideal_bond_angles = torch.deg2rad(self.ideal_bond_angles[sequence_indices][:, chi_number].unsqueeze(-1)) # type: ignore
        new_chi_angles = torch.deg2rad(chi_3_angles.unsqueeze(-1))
        next_coords = extend_coordinates(prev_coords, ideal_bond_lengths, ideal_bond_angles, new_chi_angles)

        # Update the cloned ideal coordinates with the newly computed coordinate.
        tyr_coords[torch.arange(tyr_coords.shape[0], device=tyr_coords.device), curr_indices[:, chi_number, 3]] = next_coords

        return tyr_coords

    def adjust_chi_rotatable_ideal_atom_placements(
            self, 
            ideal_coords: torch.Tensor, 
            chi_atom_indices: torch.Tensor, 
            sequence_indices: torch.Tensor, 
            predicted_atoms: torch.Tensor, 
            chi_angles: torch.Tensor
    ) -> None:
        """
        Adjusts the chi rotatable ideal atom placements.

        Args:
            ideal_coords (torch.Tensor): The ideal amino acid coordinates. (N, MAX_NUM_RESIDUE_ATOMS, 3) 
                using padded last index for easy indexing.
            chi_atom_indices (torch.Tensor): The atom indices necessary for computing the next atom location. (N, 4, 4) 
                of indices for dataset_atom_order.
            sequence_indices (torch.Tensor): The sequence indices. (N,) elements in range [0, 19]
            predicted_atoms (torch.Tensor): The predicted atoms. (N, 4) padded with MAX_NUM_RESIDUE_ATOMS
            chi_angles (torch.Tensor): The chi angles. (N, 4) padded with NaN

        Returns:
            None
        """

        index_tensor = torch.arange(ideal_coords.shape[0], device=ideal_coords.device)
        for chi_i in range(4):
            # Pull the atom indices necessary for computing the current next atom location.
            # Index the ideal amino acid coordinates with curr_indices to get the current atom locations.
            curr_indices = chi_atom_indices[:, chi_i, :3]

            # (N x 3 x 3) coordinates of the ideal residue that we will use to compute the next atom location.
            prev_atoms = ideal_coords.gather(1, curr_indices.unsqueeze(-1).expand(-1, -1, 3))

            # Get ideal bond lengths, bond angles for current chi angle index:
            ideal_bond_lengths = self.ideal_bond_lengths[sequence_indices][:, chi_i].unsqueeze(-1) # type: ignore
            ideal_bond_angles = torch.deg2rad(self.ideal_bond_angles[sequence_indices][:, chi_i].unsqueeze(-1)) # type: ignore

            # Convert dihedral angle to radians.
            new_chi_angles = torch.deg2rad(chi_angles[:, chi_i].unsqueeze(-1))

            # Compute the next atom location.
            next_coords = extend_coordinates(prev_atoms, ideal_bond_lengths, ideal_bond_angles, new_chi_angles)

            # Update the cloned ideal coordinates with the newly computed coordinate.
            ideal_coords[index_tensor, predicted_atoms[:, chi_i], :] = next_coords

    def build_rotamers(
        self, 
        backbone_coords: torch.Tensor, 
        chi_angles: torch.Tensor, 
        sequence_indices: torch.Tensor, 
        backbone_alignment_matrices: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Builds full atom coordinates given backbone coordinates, sequence labels, and chi angles.
        First rotates and aligns ideal amino acid atoms to the desired rotamer.
        Then, aligns adjusted ideal amino acids to the backbone coordinates.
            
        Args:
            backbone_coords (torch.Tensor): Tensor containing backbone coordinates. (N, 5, 3) 
                where middle index is in order (N, CA, CB, C, O)
            chi_angles (torch.Tensor): Tensor containing chi angles. (N, 4)
            sequence_indices (torch.Tensor): Tensor containing sequence labels. (N,) where each element is in range [0, 19]

            backbone_alignment_matrices (Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]): Optional 
                tuple containing backbone alignment matrices, computed if not available.
        
        Returns:
            torch.Tensor: Tensor containing the full atom coordinates. (N, MAX_NUM_RESIDUE_ATOMS, 3)
        """

        # Create ideal atomic coordiantes for each residue and make a copy which we will use to compute alignment.
        ideal_coords = self.ideal_aa_coords[sequence_indices] # type: ignore
        ideal_coords_clone = ideal_coords.clone()
        chi_atom_indices_expanded = self.aa_to_chi_angle_atom_index[sequence_indices] # type: ignore
        predicted_atom = chi_atom_indices_expanded[:, :, -1]

        # Modifies ideal_coords_expanded in place to contain the (partially) rotated coordinates.
        self.adjust_chi_rotatable_ideal_atom_placements(ideal_coords, chi_atom_indices_expanded, sequence_indices, predicted_atom, chi_angles)

        # Lookup precomputed indices with which to compute a rigid body alignment between the adjusted ideal coordinates and leftover atoms.
        leftover_atom_alignment_indices = self.alignment_indices[sequence_indices] # type: ignore
        # Lookup the indices of the atoms that we will overwrite the un-adjusted ideal coordinates with.
        leftover_mobile_indices = self.leftover_atom_indices[sequence_indices] # type: ignore

        # Gather the fixed and mobile coordinates for the alignment.
        expanded_alignment_indices = leftover_atom_alignment_indices.unsqueeze(-1).expand(-1, -1, 3)
        fixed_coords = ideal_coords.gather(1, expanded_alignment_indices)
        mobile_coords = ideal_coords_clone.gather(1, expanded_alignment_indices) # type: ignore

        # Gather the leftover coordinates that don't get updated by the adjust_chi_rotatable_ideal_atom_placements function.
        leftover_mobile_coords = ideal_coords_clone.gather(1, leftover_mobile_indices.unsqueeze(-1).expand(-1, -1, 3))

        # Align the adjusted ideal coordinates to the backbone coordinates.
        leftover_mobile_coords = apply_transformation(leftover_mobile_coords, *compute_alignment_matrices(fixed_coords, mobile_coords))

        # Update the adjusted ideal coordinates with the leftover mobile coordinates.
        # Exclude the padding index from the update.
        not_padding_index_mask = (leftover_mobile_indices != MAX_NUM_RESIDUE_ATOMS)
        row_indices, col_indices = not_padding_index_mask.nonzero(as_tuple=True)
        atom_indices = leftover_mobile_indices[not_padding_index_mask]
        ideal_coords[row_indices, atom_indices] = leftover_mobile_coords[row_indices, col_indices]

        # Run the extension algorithm using aligned residues to adjust the hydrogens placed on tyrosine.
        is_tyr_mask = sequence_indices == aa_short_to_idx['Y']
        tyr_coords = self.place_tyr_hydrogens(ideal_coords[is_tyr_mask], sequence_indices[is_tyr_mask], chi_angles[is_tyr_mask, 2])
        ideal_coords[is_tyr_mask] = tyr_coords

        # Align the completely transformed ideal coordinates to the backbone coordinates.
        fixed_backbone_coords = backbone_coords[:, self.backbone_frame_mask_1]
        mobile_backbone_coords = ideal_coords[:, :3]
        if backbone_alignment_matrices is None:
            backbone_alignment_matrices = compute_alignment_matrices(fixed_backbone_coords, mobile_backbone_coords)

        # Overwrite the aligned backbone coordinates with the true backbone coordinates to avoid only using ideal phi/psi angles.
        ideal_coords = apply_transformation(ideal_coords, *backbone_alignment_matrices)
        ideal_coords[:, :4] = backbone_coords[:, self.backbone_frame_mask_2]

        return ideal_coords


def compute_alignment_matrices(fixed: torch.Tensor, mobile: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the rotation and translation matrices that align the mobile coordinates to the fixed coordinates.
    """
    # Compute center of mass of fixed and mobile coordinate lists.
    fixed_coords_com = fixed.nanmean(dim=1, keepdim=True)
    mob_coords_com = mobile.nanmean(dim=1, keepdim=True)

    # Center the fixed and mobile coordinate lists.
    mob_coords_cen = mobile - mob_coords_com
    targ_coords_cen = fixed - fixed_coords_com

    # Compute the transformation that minimizes the RMSD between the fixed and mobile coordinate lists.
    C = torch.nan_to_num(mob_coords_cen).transpose(1, 2) @ torch.nan_to_num(targ_coords_cen)
    U, S, Wt = torch.linalg.svd(C)
    R = U @ Wt
    neg_det_mask = torch.linalg.det(R) < 0.0
    Wt[neg_det_mask, -1] *= -1
    R[neg_det_mask] = U[neg_det_mask] @ Wt[neg_det_mask]

    return R, mob_coords_com, fixed_coords_com


def apply_transformation(coords: torch.Tensor, R: torch.Tensor, mob_coords_com: torch.Tensor, fixed_coords_com: torch.Tensor) -> torch.Tensor:
    """
    Apply rotation and translation matrices computed in compute_alignment_matrices to coords.
    """
    return ((coords - mob_coords_com) @ R) + fixed_coords_com


def extend_coordinates(prev_atom_coords: torch.Tensor, bond_lengths: torch.Tensor, bond_angles: torch.Tensor, dihedral_angles: torch.Tensor) -> torch.Tensor:
    """
    Extends the coordinates of a molecule based on bond lengths, bond angles, and dihedral angles.

    Args:
        coords (torch.Tensor): A (N, 3, 3) coodinate tensor.
        bond_lengths (torch.Tensor): (N, 1) The ideal length of bond of the atom to be added.
        bond_angles (torch.Tensor): (N, 1) The ideal bond angle of atom being added.
        dihedral_angles (torch.Tensor): (N, 1) The dihedral angles being added.

    Returns:
        torch.Tensor: The (N, 3) coordinates of the fourth atom defining the desired dihedral 
            angle with the ideal bond length and bond angle.
    """
    bc = prev_atom_coords[:, 1] - prev_atom_coords[:, 2]
    bc = bc / torch.linalg.vector_norm(bc, dim=-1, keepdim=True)
    ba = torch.cross(prev_atom_coords[:, 1] - prev_atom_coords[:, 0], bc)
    ba = ba / torch.linalg.vector_norm(ba, dim=-1, keepdim=True)
    m1 = torch.cross(ba, bc)
    d1 = bond_lengths * torch.cos(bond_angles)
    d2 = bond_lengths * torch.sin(bond_angles) * torch.cos(dihedral_angles)
    d3 = -1 * bond_lengths * torch.sin(bond_angles) * torch.sin(dihedral_angles)
    next_coords = prev_atom_coords[:, 2] + (bc * d1) + (m1 * d2) + (ba * d3)
    return next_coords

