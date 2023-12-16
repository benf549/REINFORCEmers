import os
import torch
import torch.nn.functional as F
import prody as pr
from collections import defaultdict

from utils.protein_dataset import parse_pdb, compute_chi_angles, parse_canonical_prody_residue
from utils.build_rotamers import RotamerBuilder
from utils.constants import aa_long_to_short, aa_idx_to_short, aa_to_chi_angle_atom_index, dataset_atom_order, aa_idx_to_short, aa_idx_to_long, aa_short_to_idx

def create_prody_protein_from_coordinate_matrix(full_protein_coords, amino_acid_labels):
    # Drop NaNs from the coordinate matrix and flatten it
    # flattened_coords = full_protein_coords[~full_protein_coords.isnan().any(dim=-1)].cpu()
    amino_acid_indices = amino_acid_labels.cpu().tolist()

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
        all_coords.append(coord)
    flattened_coords = torch.cat(all_coords, dim=0)

    assert(all([len(x) == len(flattened_coords) for x in prody_features.values()]))
    
    protein = pr.AtomGroup('LASErMPNN Generated Protein')
    protein.setCoords(flattened_coords)
    protein.setNames(prody_features['atom_labels']) # type: ignore
    protein.setResnames(prody_features['resnames']) # type: ignore
    protein.setResnums(prody_features['resnums']) # type: ignore
    protein.setChids(prody_features['chains']) # type: ignore
    protein.setOccupancies(prody_features['occupancies'])# type: ignore

    return protein


def construct_ideal_coord_tensors() -> None:
    output_list = []
    path = './files/ideal_aas/'
    for i in os.listdir(path):
        if i.startswith('ideal'):
            aa_path = os.path.join(path, i)
            three_letter_code = i.rsplit('_', 1)[1].replace('.pdb', '').upper()

            aa_atom_group = pr.parsePDB(aa_path)
            if not type(aa_atom_group) is pr.AtomGroup:
                raise NotImplementedError

            tup = parse_canonical_prody_residue(aa_atom_group, three_letter_code)
            output_list.append(tup)
    
    sorted_outputs = sorted(output_list, key=lambda x: x[1])
    output = torch.stack([x[0] for x in sorted_outputs])
    torch.save(output, './files/new_ideal_coords.pt')


def compute_ideal_bond_lengths() -> None:
    ideal_coords = torch.load('./files/new_ideal_coords.pt')
    ideal_coords = F.pad(ideal_coords, (0, 0, 0, 1, 0, 0), 'constant', torch.nan)
    output = torch.full((20, 4), torch.nan)

    for idx in range(20):
        ideal_aa_coords = ideal_coords[idx]
        for jdx, chi_indices in enumerate(aa_to_chi_angle_atom_index[idx]):
            ideal_distance = torch.cdist(ideal_aa_coords[chi_indices[-2]].unsqueeze(0), ideal_aa_coords[chi_indices[-1]].unsqueeze(0)).flatten()
            output[idx, jdx] = ideal_distance
    torch.save(output, './files/new_ideal_bond_lengths.pt')


def compute_ideal_bond_angles() -> None:
    ideal_coords = torch.load('./files/new_ideal_coords.pt')
    ideal_coords = F.pad(ideal_coords, (0, 0, 0, 1, 0, 0), 'constant', torch.nan)
    output = torch.full((20, 4), torch.nan)
    for idx in range(20):
        ideal_aa_coords = ideal_coords[idx]
        for jdx, chi_indices in enumerate(aa_to_chi_angle_atom_index[idx]):
            a, b, c = [ideal_aa_coords[x] for x in chi_indices[-3:]]
            d = a - b
            e = c - b
            angle = torch.rad2deg(torch.acos(torch.dot(d, e) / (torch.norm(d) * torch.norm(e))))
            output[idx, jdx] = angle
    torch.save(output, './files/new_ideal_bond_angles.pt')


def main():
    device = torch.device('cpu')

    # Initialize rotamer builder.
    rotamer_builder = RotamerBuilder().to(device)

    # Compute ideal bond lengths and angles.
    construct_ideal_coord_tensors()
    compute_ideal_bond_lengths()
    compute_ideal_bond_angles()

    # Load an example protein
    # prody_protein = pr.parsePDB('./1a12_1.pdb')
    # if not type(prody_protein) is pr.AtomGroup:
    #     return

    # # Parse the protein data
    # protein_data = parse_pdb(prody_protein.getHierView())

    # # Loop over chains and compute chi angles.
    # for chain in protein_data.all_chains:
    #     chi_angles = compute_chi_angles(chain.residue_coords, chain.sequence_indices)
    #     chain.set_chi_angles(chi_angles)

    #     backbone_coords = chain.get_backbone_coords()
    #     placed_coords = rotamer_builder.build_rotamers(backbone_coords, chi_angles, chain.sequence_indices)

    #     is_x_mask = chain.sequence_indices == aa_short_to_idx['X']
    #     A = create_prody_protein_from_coordinate_matrix(chain.residue_coords[~is_x_mask], chain.sequence_indices[~is_x_mask])
    #     pr.writePDB('./A.pdb', A)
    #     B = create_prody_protein_from_coordinate_matrix(placed_coords, chain.sequence_indices)
    #     pr.writePDB('./B.pdb', B)

        # print(torch.isclose(chain.residue_coords.double(), placed_coords, rtol=0.2, equal_nan=True).all(dim=1).all(dim=1))
        # raise NotImplementedError


if __name__ == "__main__":
    main()
