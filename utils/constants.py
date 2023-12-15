import os
import torch
from pathlib import Path

parent_path = Path(__file__).parent.absolute()

# Define distance that will be used to create a mask for residues.
EXTRA_ATOM_CONTACT_DISTANCE = 5.0
ON_TO_S_HBOND_MAX_DISTANCE = 4.2
S_TO_S_HBOND_MAX_DISTANCE = 4.5
HBOND_MAX_DISTANCE = 3.5
ATOM_IDENTITY_ENUM = ['C', 'O', 'N', 'S', 'metal', 'other', 'Padding']
HBOND_CAPABLE_ELEMENTS = ('N', 'O', 'S', 'Padding')
DISULFIDE_S_CLASH_DIST = 1.8
OTHER_ATOM_CLASH_DIST = 2.7
METAL_CLASH_DIST = 1.5
MAX_PEPTIDE_LENGTH = 40
NUM_CB_ATOMS_FOR_BURIAL = 16
CHI_BIN_MIN, CHI_BIN_MAX = -180, 180
HARD_CLASH_TOLERANCE = 0.2

# Map of canonical amino acid 1 to 3 letter codes.
aa_short_to_long = {'C': 'CYS', 'D': 'ASP', 'S': 'SER', 'Q': 'GLN', 'K': 'LYS', 'I': 'ILE', 'P': 'PRO', 'T': 'THR', 'F': 'PHE', 'N': 'ASN', 'G': 'GLY', 'H': 'HIS', 'L': 'LEU', 'R': 'ARG', 'W': 'TRP', 'A': 'ALA', 'V': 'VAL', 'E': 'GLU', 'Y': 'TYR', 'M': 'MET', 'X': 'XAA'}
aa_long_to_short = {x: y for y, x in aa_short_to_long.items()}
aa_long_to_idx = {'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4, 'GLU': 5, 'GLN': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9, 'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14, 'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19, 'XAA': 20}
aa_short_to_idx = {x: aa_long_to_idx[y] for x, y in aa_short_to_long.items()}
aa_idx_to_long = {x: y for y, x in aa_long_to_idx.items()}
aa_idx_to_short = {x: aa_long_to_short[y] for x, y in aa_idx_to_long.items()}

clash_matrix = torch.tensor([
    #  C,   O,   N,   S, metals, other
    [3.4, 3.0, 3.0, 3.4, METAL_CLASH_DIST, OTHER_ATOM_CLASH_DIST, 0], # C
    [3.0, 2.5, 2.5, 3.0, METAL_CLASH_DIST, OTHER_ATOM_CLASH_DIST, 0], # O
    [3.0, 2.5, 2.5, 3.0, METAL_CLASH_DIST, OTHER_ATOM_CLASH_DIST, 0], # N
    [3.4, 3.0, 3.0, 3.4, METAL_CLASH_DIST, OTHER_ATOM_CLASH_DIST, 0], # S
    [METAL_CLASH_DIST] * 6 + [0],                                        # metals
    ([OTHER_ATOM_CLASH_DIST]*4) + [METAL_CLASH_DIST, OTHER_ATOM_CLASH_DIST, 0], # other
    [0] * 7,                                        # metals
])

# Map of one letter amino acid codes to their corresponding atom order.
dataset_atom_order = {
    'G': ['N', 'CA', 'C', 'O'],
    'X': ['N', 'CA', 'C', 'O'],
    'A': ['N', 'CA', 'C', 'O', 'CB'],
    'S': ['N', 'CA', 'C', 'O', 'CB', 'OG', 'HG'],
    'C': ['N', 'CA', 'C', 'O', 'CB', 'SG', 'HG'],
    'T': ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2', 'HG1'],
    'P': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD'],
    'V': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2'],
    'M': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'SD', 'CE'],
    'N': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'ND2'],
    'I': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1'],
    'L': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2'],
    'D': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'OD2'],
    'E': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'OE2'],
    'K': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ'],
    'Q': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'NE2'],
    'H': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2', 'HD1', 'HE2'],
    'F': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
    'R': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2'],
    'Y': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH', 'HH'],
    'W': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE2', 'CE3', 'NE1', 'CZ2', 'CZ3', 'CH2']
}

# The maximum number of atoms for a single residue.
MAX_NUM_RESIDUE_ATOMS = max([len(res) for res in dataset_atom_order.values()])

# Map from chi angle index to the atoms that define it. Includes rotatable hydrogen chi angles.
aa_to_chi_angle_atom_map = {
    'C': {1: ('N', 'CA', 'CB', 'SG'), 2: ('CA', 'CB', 'SG', 'HG')},
    'D': {1: ('N', 'CA', 'CB', 'CG'), 2: ('CA', 'CB', 'CG', 'OD1')},
    'E': {1: ('N', 'CA', 'CB', 'CG'), 2: ('CA', 'CB', 'CG', 'CD'), 3: ('CB', 'CG', 'CD', 'OE1')},
    'F': {1: ('N', 'CA', 'CB', 'CG'), 2: ('CA', 'CB', 'CG', 'CD1')},
    'H': {1: ('N', 'CA', 'CB', 'CG'), 2: ('CA', 'CB', 'CG', 'ND1')},
    'I': {1: ('N', 'CA', 'CB', 'CG1'), 2: ('CA', 'CB', 'CG1', 'CD1')},
    'K': {1: ('N', 'CA', 'CB', 'CG'), 2: ('CA', 'CB', 'CG', 'CD'), 3: ('CB', 'CG', 'CD', 'CE'), 4: ('CG', 'CD', 'CE', 'NZ')},
    'L': {1: ('N', 'CA', 'CB', 'CG'), 2: ('CA', 'CB', 'CG', 'CD1')},
    'M': {1: ('N', 'CA', 'CB', 'CG'), 2: ('CA', 'CB', 'CG', 'SD'), 3: ('CB', 'CG', 'SD', 'CE')},
    'N': {1: ('N', 'CA', 'CB', 'CG'), 2: ('CA', 'CB', 'CG', 'OD1')},
    'P': {1: ('N', 'CA', 'CB', 'CG'), 2: ('CA', 'CB', 'CG', 'CD')},
    'Q': {1: ('N', 'CA', 'CB', 'CG'), 2: ('CA', 'CB', 'CG', 'CD'), 3: ('CB', 'CG', 'CD', 'OE1')},
    'R': {1: ('N', 'CA', 'CB', 'CG'), 2: ('CA', 'CB', 'CG', 'CD'), 3: ('CB', 'CG', 'CD', 'NE'), 4: ('CG', 'CD', 'NE', 'CZ')},
    'S': {1: ('N', 'CA', 'CB', 'OG'), 2: ('CA', 'CB', 'OG', 'HG')},
    'T': {1: ('N', 'CA', 'CB', 'OG1'), 2: ('CA', 'CB', 'OG1', 'HG1')},
    'V': {1: ('N', 'CA', 'CB', 'CG1')},
    'W': {1: ('N', 'CA', 'CB', 'CG'), 2: ('CA', 'CB', 'CG', 'CD1')},
    # NOTE: will need to align leftover atoms to the first two chi angles before the final chi angle only for TYR.
    'Y': {1: ('N', 'CA', 'CB', 'CG'), 2: ('CA', 'CB', 'CG', 'CD1'), 3: ('CE1', 'CZ', 'OH', 'HH')}
}

##### Converts the atom names defined above to indices in the dataset atom order tensor
#### Assumes tensors are padded with with NaN at MAX_NUM_RESIDUE_ATOMS'th index in dim 1 
placeholder_indices = torch.tensor([MAX_NUM_RESIDUE_ATOMS] * 4)
aa_to_chi_angle_atom_index = torch.full((20, 4, 4), MAX_NUM_RESIDUE_ATOMS)
aa_to_chi_angle_mask = torch.full((20, 4), False)
aa_to_leftover_atoms = torch.full((20, MAX_NUM_RESIDUE_ATOMS), MAX_NUM_RESIDUE_ATOMS)
# Iterate in the order of the canonical amino acid indices in aa_idx_to_short
for idx in range(20):
    aa = aa_idx_to_short[idx]
    if aa in aa_to_chi_angle_atom_map:
        # Fill residues that have chi angles with indices of relevant atoms.
        all_atoms_set = set([x for x in range(len(dataset_atom_order[aa]))])
        chi_placed_atoms_set = set()
        for chi_num, atom_names in aa_to_chi_angle_atom_map[aa].items():
            chi_placed_indices = [dataset_atom_order[aa].index(x) for x in atom_names]
            aa_to_chi_angle_atom_index[idx, chi_num - 1] = torch.tensor(chi_placed_indices)
            chi_placed_atoms_set.update(chi_placed_indices)

        # Track which atoms are not involved in the placement process
        leftovers = sorted(list(all_atoms_set - chi_placed_atoms_set - {2, 3}))
        aa_to_leftover_atoms[idx, :len(leftovers)] = torch.tensor(leftovers)
        
        # Fill mask with True for chi angles and False for padding.
        aa_to_chi_angle_mask[idx, :len(aa_to_chi_angle_atom_map[aa])] = True

# Remove the terminal TYR chi angle atoms that we won't actually place when adjusting angles.
tyr_idx = aa_short_to_idx['Y']
num_tyr_leftover = (aa_to_leftover_atoms[tyr_idx] != MAX_NUM_RESIDUE_ATOMS).sum().item()
tyr_leftover_updated = sorted(list(set(aa_to_leftover_atoms[tyr_idx, :num_tyr_leftover].tolist() + [dataset_atom_order['Y'].index(y) for y in ['CE1', 'CZ', 'OH', 'HH']])))
aa_to_leftover_atoms[tyr_idx, :len(tyr_leftover_updated)] = torch.tensor(tyr_leftover_updated)
aa_to_leftover_atoms = aa_to_leftover_atoms.narrow(dim=1, start=0, length=(aa_to_leftover_atoms != 14).sum(dim=1).max()) # type: ignore

# Load precomputed ideal coordinates, bond lengths, and bond angles necessary to build rotamers. 
# Computed from idealized single amino acid PDB files generated by something like Rosetta.
ideal_aa_coords = torch.load(os.path.join(parent_path, '../files', 'new_ideal_coords.pt'))
ideal_bond_lengths = torch.load(os.path.join(parent_path, '../files', 'new_ideal_bond_lengths.pt'))
ideal_bond_angles = torch.load(os.path.join(parent_path, '../files', 'new_ideal_bond_angles.pt'))
alignment_indices = torch.load(os.path.join(parent_path, '../files', 'rotamer_alignment.pt'))
alignment_indices_mask = alignment_indices == -1
alignment_indices[alignment_indices_mask] = MAX_NUM_RESIDUE_ATOMS

amino_acid_to_atom_identity_matrix = torch.zeros(20, 15, dtype=torch.long)
for aa, atom_list in dataset_atom_order.items():
    if aa == 'X':
        continue
    amino_acid_to_atom_identity_matrix[aa_long_to_idx[aa_short_to_long[aa]]] = torch.tensor([ATOM_IDENTITY_ENUM.index(atom_list[idx][0]) if idx < idx < len(atom_list) and atom_list[idx][0] in ATOM_IDENTITY_ENUM else ATOM_IDENTITY_ENUM.index("Padding") for idx in range(15)])

hbond_candidate_indices = torch.tensor([aa_long_to_idx[aa_short_to_long[x]] for x,y in dataset_atom_order.items() if any(atom[0] in HBOND_CAPABLE_ELEMENTS for atom in y[4:])])
hbond_candidate_set = {aa_idx_to_short[x] for x in hbond_candidate_indices.tolist()}

hbond_mask_dict = {}
hbond_element_dict = {}
for residue in hbond_candidate_set:
    residue_index_list = []
    residue_element_list = []
    for idx, atom in enumerate(dataset_atom_order[residue][4:]):
        # Ignore non-{O, N, S} atoms.
        if not atom[0] in HBOND_CAPABLE_ELEMENTS:
            continue

        # Store index and element in the same orders.
        residue_index_list.append(idx)
        residue_element_list.append(HBOND_CAPABLE_ELEMENTS.index(atom[0]))

    # Store indices for atoms in coordinate matrices and elements for each atom.
    hbond_mask_dict[residue] = torch.tensor(residue_index_list)
    hbond_element_dict[residue] = residue_element_list
