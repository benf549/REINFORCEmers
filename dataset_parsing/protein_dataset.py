import prody
import torch
from torch_cluster import radius_graph
from torch_scatter import scatter
import torch.nn.functional as F

from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Optional, Tuple, List, Union, Dict
from collections import defaultdict
from .constants import MAX_NUM_RESIDUE_ATOMS, EXTRA_ATOM_CONTACT_DISTANCE, dataset_atom_order, aa_short_to_long, aa_long_to_short, aa_long_to_idx, aa_short_to_idx, aa_to_chi_angle_atom_index, aa_idx_to_short

@dataclass
class ChainData:
    residue_coords: torch.Tensor
    resnums: List[int]
    sequence_indices: torch.Tensor # Amino acid sequence numerical encodings.
    pdb_code: str

    remapped_chain_id: Tuple[str, str] #segment_id, chain_id

    original_chain_id: Union[Tuple[str, str], List[Tuple[str, str]], None] = None
    chi_angles: Optional[torch.Tensor] = None
    seqres_resnums: Optional[List[Union[int, None]]] = None
    multiple_sequence_alignment: Optional[torch.Tensor] = None
    backbone_coords: Optional[torch.Tensor] = None
    
    extra_atom_contact_mask: Optional[torch.Tensor] = None
    cb_residue_counts: Optional[torch.Tensor] = None
    polymer_seq: Optional[str] = None

    def is_empty(self):
        """Check if there is any parsed data in this chain."""
        return self.residue_coords.shape[0] == 0
    
    def set_chi_angles(self, chi_angles):
        # Sanity check that chi angles are the same shape as the residue coordinates.
        A = chi_angles.shape[0] 
        B = self.residue_coords.shape[0]
        if A != B:
            raise ValueError(f"Missing chi angles for some residues. First dimenensions are {A} != {B}")

        self.chi_angles = chi_angles

    def set_backbone_coords(self, backbone_coords) -> None:
        self.backbone_coords = backbone_coords

    def get_backbone_coords(self) -> torch.Tensor:
        """Outputs the backbone coordinates in order [N, Ca, Cb, C, O]."""
        if self.backbone_coords is None:
            backbone_coords = self.residue_coords.gather(1, torch.tensor([[0, 1, 4, 2, 3]]).unsqueeze(-1).expand(self.residue_coords.shape[0], -1, 3))
            self.set_backbone_coords(backbone_coords)
            return backbone_coords
        return self.backbone_coords

    def set_original_chain_id(self, chain_id_mapping: Union[Tuple[str, str], List[Tuple[str, str]]]) -> None:
        self.original_chain_id = chain_id_mapping
    
    def set_seqres_resnums(self, seqres_resnums: List[Union[int, None]]) -> None:
        # Sanity check that the MSA data matches the polymer sequence.
        assert(len(seqres_resnums) == self.residue_coords.shape[0])

        self.seqres_resnums = seqres_resnums

    def set_multiple_sequence_alignment(self, msa_data: torch.Tensor) -> None:
        if self.seqres_resnums is None:
            raise NotImplementedError("Cannot set MSA data without seqres resnums.")
    
        # Use the seqres_resnums to index the relevant rows of the MSA data.
        index_tensor = torch.tensor(self.seqres_resnums) - 1
        self.multiple_sequence_alignment = msa_data[index_tensor]

    def set_multiple_sequence_alignment_from_singletons(self, chain_singleton_msa_data: List[torch.Tensor], seqres_list: List[int]) -> None:
        if self.seqres_resnums is None:
            raise NotImplementedError("Cannot set MSA data without seqres resnums.")
        
        default_msa_row = torch.zeros(21)

        out = []
        for seq_res, msa_row in zip(seqres_list, chain_singleton_msa_data):
            if seq_res is None:
                out.append(default_msa_row)
            else:
                out.append(msa_row)
        self.multiple_sequence_alignment = torch.stack(out)

    def set_extra_atom_contact_mask(self, extra_atom_coords: torch.Tensor) -> None:
        contact_mask = torch.cdist(self.residue_coords.double(), extra_atom_coords.double()) < EXTRA_ATOM_CONTACT_DISTANCE
        self.extra_atom_contact_mask = contact_mask.any(dim=1).any(dim=1)
    
    def set_cb_counts(self, cb_counts: torch.Tensor) -> None:
        self.cb_residue_counts = cb_counts
    
    def get_virtual_cbeta_coords(self) -> torch.Tensor:
        backbone_coords = self.get_backbone_coords()
        gly_residues = torch.isin(self.sequence_indices, torch.tensor([aa_short_to_idx['G'], aa_short_to_idx['X']]))

        # Only operate on glycine residues.
        gly_residue_coords = backbone_coords[gly_residues]
        if gly_residue_coords.shape[0] == 0:
            return backbone_coords

        # Compute the virtual C-beta coordinates for glycine residues.
        b = gly_residue_coords[:, 1] - gly_residue_coords[:, 0] # CA - N
        c = gly_residue_coords[:, 3] - gly_residue_coords[:, 1] # C - CA
        a = torch.cross(b, c, dim=-1)
        gly_residue_coords[:, 2] = -0.58273431*a + 0.56802827*b - 0.54067466*c + gly_residue_coords[:, 1]
        backbone_coords[gly_residues] = gly_residue_coords.clone()

        return backbone_coords
    
    def set_polymer_seq(self, polymer_seq) -> None:
        self.polymer_seq = polymer_seq
    
    def to_output_dict(self) -> Dict[str, torch.Tensor]:

        if self.extra_atom_contact_mask is None or self.cb_residue_counts is None or self.chi_angles is None:
            raise NotImplementedError

        # Dump chain data into a dictionary.
        output = {
            'backbone_coords': self.get_virtual_cbeta_coords().float(),
            'chi_angles': self.chi_angles.float(),
            'sequence_indices': self.sequence_indices.type(torch.uint8),

            'msa_data': self.multiple_sequence_alignment,
            'extra_atom_contact_mask': self.extra_atom_contact_mask,
            'residue_cb_counts': self.cb_residue_counts.type(torch.uint8),

            'size': self.residue_coords.shape[0],
            'polymer_seq': self.polymer_seq,
        }

        # Sanity check that all tensors have the same first dimension.
        size = output['size']
        for v in output.values():
            if type(v) is torch.Tensor:
                assert(v.shape[0] == size)

        return output

        
    
@dataclass
class ExtraAtomInfo:
    """
    Stores the coordinates of extra atoms that we would otherwise discard.
    Use this information to generate masks for what atoms we should use for training.
    """
    # Coordinates of all extra atoms.
    all_extra_atom_coords: torch.Tensor
    # TODO: implmenet vdW-radius/dynamic atom distance contact cutoffs.
    all_extra_atom_elements: List[str]


class ProteinIsEmptyError(Exception):
    pass


@dataclass
class ProteinData:
    """
    Stores the chain data.
    """
    all_chains: List[ChainData]
    extra_atom_info: ExtraAtomInfo
    pdb_code: str

    def compute_residue_num_cb_contacts(self, cb_distance_for_burial_calculation=10) -> None:
        # Construct a tensor with all the coordinates.
        all_backbone_coords = []
        chain_index_tensor = []
        for idx, chain in enumerate(self.all_chains):
            all_backbone_coords.append(chain.get_virtual_cbeta_coords())
            chain_index_tensor.extend([idx] * chain.residue_coords.shape[0])
        
        if len(all_backbone_coords) == 0:
            raise ProteinIsEmptyError("There are no residues in this protein.")

        all_backbone_coords = torch.cat(all_backbone_coords, dim=0)
        all_residue_chain_indices = torch.tensor(chain_index_tensor)

        # Compute C-beta counts.
        connected_cb_edge_graph = radius_graph(all_backbone_coords[:, 2], cb_distance_for_burial_calculation, max_num_neighbors=100)

        # Num CB atoms within cb_distance_for_burial_calculation per residue.
        residue_counts = connected_cb_edge_graph[1].bincount(minlength=all_backbone_coords.shape[0])

        # Set the CB counds for each chain.
        for idx in range(len(self.all_chains)):
            self.all_chains[idx].set_cb_counts(residue_counts[idx == all_residue_chain_indices])

    def generate_final_data_dict_for_serialization(self) -> Dict[str, torch.Tensor]:

        output = {}

        if len(self.all_chains) == 0:
            return output

        try:
            self.compute_residue_num_cb_contacts()
        except ProteinIsEmptyError:
            print(self.all_chains[0].pdb_code, 'is empty.')
            return output

        for chain in self.all_chains:
            chain.set_extra_atom_contact_mask(self.extra_atom_info.all_extra_atom_coords)
            output[chain.remapped_chain_id] = chain.to_output_dict()
        
        return output



def get_padded_residue_coordinate_tensor():
    """
    Returns a tensor of shape (MAX_NUM_ATOMS, 3) filled with NaNs.
    """
    return torch.full((MAX_NUM_RESIDUE_ATOMS, 3), torch.nan)


def parse_generic_prody_residue(prody_residue) -> Tuple[torch.Tensor, List[str]]:
    """
    Function handles any arbitrary ProDy residue object and returns a tuple of the atom coordiantes and elements.
    """
    atom_coords = []
    atom_elements = []

    # Loop over atoms, coords, and elements.
    for _atom, coord, elem in zip(prody_residue.getNames(), prody_residue.getCoords(), prody_residue.getElements()):
        # Ignore hydrogens.
        if elem == 'H':
            continue

        # Save the heavy atoms.
        atom_coords.append(torch.tensor(coord))
        atom_elements.append(elem)

    return torch.stack(atom_coords), atom_elements


def noncanonical_parse_canonical_prody_residue(prody_residue):
    """
    Analogous to parse_canonical_prody_residue but for non-canonical residues.

    Args:
        prody_residue (ProDy residue): The ProDy residue object to convert.

    Returns :
        tuple: A tuple containing the residue coordinate tensor and the sequence amino acid index representation of the residue name,
            a tensor of the the remaining heavy extra atom coordinates, and a list of the extra atom elements.
    """
    extra_atom_coords = []
    extra_atom_elements = []
    residue_tensor = get_padded_residue_coordinate_tensor()
    for atom, coord, elem in zip(prody_residue.getNames(), prody_residue.getCoords(), prody_residue.getElements()):
        if elem == 'H':
            # Ignore hydrogens for ncAAs.
            continue
        elif atom in ['N', 'CA', 'C', 'O']:
            # NOTE: Previously checked that these atoms are present in residue.
            # Save the backbone atoms.
            residue_tensor[dataset_atom_order['X'].index(atom)] = torch.tensor(coord)
        else:
            # Save the extra atoms.
            extra_atom_coords.append(torch.tensor(coord))
            extra_atom_elements.append(elem)
    
    if extra_atom_coords:
        extra_atom_coords = torch.stack(extra_atom_coords)
    else:
        extra_atom_coords = torch.empty(0, 3, dtype=torch.float)
    
    
    return residue_tensor, torch.tensor(aa_short_to_idx['X'], dtype=torch.long), extra_atom_coords, extra_atom_elements


def parse_canonical_prody_residue(prody_residue, resname: str) -> Tuple[torch.Tensor, torch.Tensor]: 
    """
    Converts a ProDy residue object to a coordinate tensor.

    Args:
        prody_residue (prody.Residue): The ProDy residue object to convert.
        resname (str): The (3 letter code) name of the residue.

    Returns:
        tuple: A tuple containing the coordinate tensor and the sequence amino acid index representation of the residue name.
    """
    # Grab tensor that will store the coordinates of each atom in the residue.
    residue_tensor = get_padded_residue_coordinate_tensor()

    # Get the atom order for this residue.
    res_atom_order = dataset_atom_order[aa_long_to_short[resname]]

    # Zip together the atom names and coordinates.
    for atom, coord in zip(prody_residue.getNames(), prody_residue.getCoords()):
        if atom in res_atom_order:
            residue_tensor[res_atom_order.index(atom)] = torch.tensor(coord)
    
    return residue_tensor, torch.tensor(aa_long_to_idx[resname], dtype=torch.long)
    

def check_residue_is_well_formed(coordinate_tensor: torch.Tensor, sequence_encoding: int) -> bool:
    """
    Checks if a residue was parsed with all expected atoms.

    Args:
        coordinate_tensor (torch.Tensor): The coordinate tensor of the residue.
        sequence_encoding (int): The sequence index encoding of the residue.

    Returns:
        bool: True if the residue is well-formed, False otherwise.
    """

    # Compute number of atoms in this residue if all atoms are present.
    num_atoms = len(dataset_atom_order[aa_idx_to_short[sequence_encoding]])

    # Handle histidine which does not need all attached hydrogens to be well-formed.
    if aa_idx_to_short[sequence_encoding] == 'H':
        # Hydrogens are the last two atoms in the atom order so just subtract 2 from the max num atoms.
        num_atoms -= 2

    # Handle cysteine missing hydrogen as is the case in disulfide bonds.
    if aa_idx_to_short[sequence_encoding] == 'C':
        num_atoms -= 1

    # True if none of these atoms are NaN.
    return not coordinate_tensor[:num_atoms].isnan().any().item()



def process_chain(chain: prody.Chain, chain_id: Tuple[str, str], pdb_code: str) -> Tuple[ChainData, torch.Tensor, List[str]]:
    """
    Consumes a prody hierview segment/chain object and returns a ChainData object.

    Args:
        chain (prody.Chain): The prody chain object to be processed.
        chain_id Tuple[str, str]: The ID of the chain.

    Returns:
        Tuple[ChainData, torch.Tensor, List[str]]: A tuple containing the processed chain data, 
        extra atom coordinates, and extra atom element list.
    """
    chain_residue_coords = []
    chain_sequence_indices = []
    chain_resnums = []
    extra_atom_coords = []
    extra_atom_element_list = []
    for residue in chain.iterResidues():
        resname = residue.getResname() 

        # Handle water molecules (ignore them)
        if resname == 'HOH':
            continue

        # Handle Canonical Amino Acids.
        if resname in aa_short_to_long.values() and resname != aa_short_to_long['X']:
            coordinate_tensor, sequence_encoding = parse_canonical_prody_residue(residue, resname)
            residue_is_well_formed = check_residue_is_well_formed(coordinate_tensor, int(sequence_encoding.item()))
            if residue_is_well_formed:
                # Save the residue coordinates, sequence indices, and current resnum.
                chain_residue_coords.append(coordinate_tensor)
                chain_sequence_indices.append(sequence_encoding)
                chain_resnums.append(residue.getResnum())
            else:
                resolved_backbone = False
                # Use backbone coordinates as 'X' residue if the frame is resolved.
                if not coordinate_tensor[:4].isnan().any().item():
                    output = torch.full((MAX_NUM_RESIDUE_ATOMS, 3), torch.nan, dtype=torch.float)
                    output[:4] = coordinate_tensor[:4]
                    chain_residue_coords.append(output)
                    chain_sequence_indices.append(torch.tensor(aa_short_to_idx['X'], dtype=torch.long))
                    chain_resnums.append(residue.getResnum())

                # Put remaining atoms in extra atoms tensor.
                coordinate_mask = ~coordinate_tensor.isnan().any(dim=1)
                if resolved_backbone:
                    coordinate_mask[:4] = False

                extra_coords = coordinate_tensor[coordinate_mask]
                extra_atoms = [atom_name[0] for idx, atom_name in enumerate(dataset_atom_order[aa_idx_to_short[int(sequence_encoding.item())]]) if coordinate_mask[idx].item()]
                extra_atom_coords.append(extra_coords)
                extra_atom_element_list.extend(extra_atoms)
            continue

        # Handle non-canonical amino acids (That have a canonical backbone).
        residue_ag = residue.copy()
        if all([x in set(residue_ag.getNames()) for x in ['N', 'CA', 'C', 'O']]):

            # For now just mark these all as 'X' and just record backbone coordinates.
            coordinate_tensor, sequence_encoding, atom_coords, element_list = noncanonical_parse_canonical_prody_residue(residue_ag)

            chain_residue_coords.append(coordinate_tensor)
            chain_sequence_indices.append(sequence_encoding)
            chain_resnums.append(residue.getResnum())

            extra_atom_coords.append(atom_coords)
            extra_atom_element_list.extend(element_list)
            continue

        # TODO: Handle nucleic acids, ligands, metals separately.
        atom_coords, element_list = parse_generic_prody_residue(residue)
        extra_atom_coords.append(atom_coords)
        extra_atom_element_list.extend(element_list)

    if chain_residue_coords:
        chain_residue_coords = torch.stack(chain_residue_coords)
        chain_sequence_indices = torch.stack(chain_sequence_indices)
    else:
        chain_residue_coords = torch.empty(0, MAX_NUM_RESIDUE_ATOMS, 3, dtype=torch.float)
        chain_sequence_indices = torch.empty(0, dtype=torch.long)

    if extra_atom_coords:
        extra_atom_coords = torch.cat(extra_atom_coords, dim=0)
    else:
        extra_atom_coords = torch.empty(0, 3, dtype=torch.float)
    
    # Sanity check list lengths.
    assert(chain_residue_coords.shape[0] == chain_sequence_indices.shape[0])
    assert(chain_residue_coords.shape[0] == len(chain_resnums))
    assert(extra_atom_coords.shape[0] == len(extra_atom_element_list))

    chain_data = ChainData( 
        residue_coords=chain_residue_coords,
        sequence_indices=chain_sequence_indices,
        resnums=chain_resnums,
        remapped_chain_id=chain_id,
        pdb_code=pdb_code,
    ) 

    return chain_data, extra_atom_coords, extra_atom_element_list 


def parse_pdb(prody_protein_hierview, pdb_code) -> ProteinData:
    """ 
    Convert a prody protein object into a ProteinData object. 

    Args:
        prody_protein_hierview: A prody protein object representing the protein structure.

    Returns:
        ProteinData: A custom data object containing parsed protein data.
    """

    # Loop over all chains in the protein and 
    parsed_protein_data = defaultdict(list)
    for chain in prody_protein_hierview:
        # Convert this chain into a ChainData object storing the residue coordinates and sequence indices and chain identifiers
        chain_id = (chain.getSegname(), chain.getChid())
        chain_data, extra_atom_coords, extra_atom_element_list = process_chain(chain, chain_id, pdb_code)
        if not chain_data.is_empty():
            parsed_protein_data['all_chains'].append(chain_data)
        parsed_protein_data['all_extra_atom_coords'].append(extra_atom_coords)
        parsed_protein_data['all_extra_atom_elements'].extend(extra_atom_element_list)

        # Sanity check that the number of extra atoms is the same as the number of extra atom elements.
        assert(len(extra_atom_coords) == len(extra_atom_element_list))

    # Concatenate all of the chains and extra atoms into a single tensor.
    all_extra_atom_coords = torch.cat(parsed_protein_data['all_extra_atom_coords'], dim=0)

    extra_info = ExtraAtomInfo(all_extra_atom_coords, parsed_protein_data['all_extra_atom_elements'])
    parsed_protein_data = ProteinData(parsed_protein_data['all_chains'], extra_info, pdb_code)
    return parsed_protein_data


def compute_chi_angles(coordinate_matrix: torch.Tensor, residue_indices: torch.Tensor) -> torch.Tensor:
    """
    Compute the chi angles for a given set of residue coordinates and identities.

    Args:
        coordinate_matrix (torch.Tensor): The coordinate matrix containing the (N, 14, 3) atomic coordinates.
        residue_indices (torch.Tensor dtype: long): The indices of the residues for which to compute the chi angles.

    Returns:
        torch.Tensor: The computed chi angles.
    """

    # Initialize output tensor.
    output = torch.full((coordinate_matrix.shape[0], 4), torch.nan, dtype=torch.float, device=coordinate_matrix.device)

    # Handle unknown residues (just treat them as GLY/no chi angles).
    x_residue_mask = residue_indices == aa_short_to_idx['X']
    coordinate_matrix = coordinate_matrix[~x_residue_mask]
    residue_indices = residue_indices[~x_residue_mask]

    # Handle no residues after masking for unknown residues. Without this the F.pad line below fails to handle empty tensor.
    if coordinate_matrix.shape[0] == 0:
        return output

    # Expand coordinates to (N, 15, 3) for easy indexing.
    nan_padded_coords = F.pad(coordinate_matrix, (0, 0, 0, 1, 0, 0), 'constant', torch.nan)
    expanded_sidechain_coord_indices = aa_to_chi_angle_atom_index[residue_indices]

    # Gather coordinates for chi angle computation as (N, 4, 4, 3) tensor.
    chi_coords_stacked = nan_padded_coords.gather(1, expanded_sidechain_coord_indices.flatten(start_dim=1).unsqueeze(-1).expand(-1, -1, 3)).reshape(residue_indices.shape[0], 4, 4, 3)

    # Compute batched chi-angles: 
    # https://en.wikipedia.org/w/index.php?title=Dihedral_angle&oldid=689165217#Angle_between_three_vectors
    b0 = chi_coords_stacked[:, :, 0, :] - chi_coords_stacked[:, :, 1, :]
    b1 = chi_coords_stacked[:, :, 1, :] - chi_coords_stacked[:, :, 2, :]
    b2 = chi_coords_stacked[:, :, 2, :] - chi_coords_stacked[:, :, 3, :]
    n1 = torch.cross(b0, b1)
    n2 = torch.cross(b1, b2)
    m1 = torch.cross(n1, b1 / torch.linalg.vector_norm(b1, dim=2, keepdim=True))
    x = torch.sum(n1 * n2, dim=-1)
    y = torch.sum(m1 * n2, dim=-1)

    chi_angles = torch.rad2deg(torch.arctan2(y, x))

    # Update output tensor with computed chi angles.
    output[~x_residue_mask] = chi_angles

    return output


def load_prody_hierview(path_to_protein: str) -> prody.HierView:
    """
    Construct a prody HierView object from a PDB file path.

    Raises:
        NotImplementedError: If the protein cannot be loaded or is loaded as a non-prody AtomGroup object.
    """
    
    protein = prody.parsePDB(path_to_protein) 
    if not type(protein) == prody.AtomGroup:
        if protein is None:
            raise FileNotFoundError(f"Couldn't load file {path_to_protein}")
        else:
            raise NotImplementedError("parsePDB returned a non-AtomGroup object. TODO: implement support for this.")

    return protein.getHierView()