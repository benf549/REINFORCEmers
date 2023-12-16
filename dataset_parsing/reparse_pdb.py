#!/usr/bin/env python3
"""
Given the reduce-protonated PDB files and our metadata collected so far, construct torch tensors for each protein.
See README.md for more details.

Benjamin Fry (bfry@g.harvard.edu)
"""

import os
import torch
import pickle
import prody as pr
import pandas as pd
from typing import Union
from utils.protein_dataset import parse_pdb, compute_chi_angles, load_prody_hierview, ChainData
from utils.sequence_dataset import MultipleSequenceAlignmentDataset, load_msa_from_disk, MSA_Data
from typing import Optional, Dict 
from collections import defaultdict
from tqdm import tqdm
import multiprocessing

from prepare_reduce_inputs import initialize_output_directory


def load_metadata(params: dict) -> pd.DataFrame:
    """
    Load metadata from a pickle file.

    Parameters:
        params (dict): A dictionary containing the path to the metadata file.

    Returns:
        pd.DataFrame: The loaded metadata as a pandas DataFrame.
    """
    assert(params['path_to_metadata'].endswith('.pkl'))
    return pd.read_pickle(params['path_to_metadata'])


def reformat_resnum_mapping_dict(remapped_sequence_to_seqres):
    """
    Converts dict to dict of dict mapping original chain ID to dict mapping new residue index to original residue index.
    """
    output = defaultdict(dict)
    for tup, str_val in remapped_sequence_to_seqres.items():
        if str_val.isnumeric():
            output[tup[0], tup[1]][int(str_val)] = tup[2]
    return output


def invert_dict(dict):
    return {v:k for k,v in dict.items()}


def remap_chains(
    chain: ChainData, 
    polymer_lig_dict: dict, 
    remapped_sequence_indices_to_seqres: dict, 
    residue_reindexing_map: dict, 
    all_seg_chid_map: dict, 
    msa_data: Optional[Dict[str, MSA_Data]]
) -> None:

    ### Handle residues in singleton segments.
    if chain.remapped_chain_id not in all_seg_chid_map:
        relevant_resnums = set(chain.resnums)
        remapped_chain_ids = [x for x in all_seg_chid_map.keys() if (x[:2] == chain.remapped_chain_id) if x[2] in relevant_resnums]
        old_chain_ids = [all_seg_chid_map[x] for x in remapped_chain_ids]
        old_chids = [x[1] for x in old_chain_ids]

        ### Record the MSA data in the chain object.
        if len(old_chids) == 0 or msa_data is None:
            return
        chain.set_original_chain_id(old_chids if len(old_chids) > 1 else old_chids[0])
        all_polymer_chids = {x.chid for x in polymer_lig_dict['polymers']}
        
        # Map from sequences to MSA data to find the MSA data for this chain.
        seq_to_msa_data = {x.get_query_sequence(): x for x in msa_data.values()}

        all_seqres_resnums = []
        chain_singleton_msa_data = []
        for original_chid, old_chain_id, remapped_chain_id in zip(old_chids, old_chain_ids, remapped_chain_ids):

            # Find the chid that we can use to get the MSA data.
            if '-' in original_chid and not original_chid in all_polymer_chids:
                original_chid = original_chid.split('-')[0]

            polymer_seqs = [x for x in polymer_lig_dict['polymers'] if x.chid == original_chid]

            if len(polymer_seqs) == 0 or not polymer_seqs[0].sequence in seq_to_msa_data:
                # print("Missing MSA data for chain:", original_chid)
                all_seqres_resnums.append(None)
                chain_singleton_msa_data.append(torch.zeros(21))
                continue
            polymer_seq = polymer_seqs[0].sequence

            # Make maps for the chain.
            curr_chain_msa_data = seq_to_msa_data[polymer_seq]
            fake_resnum_to_real_resnum = invert_dict(residue_reindexing_map[old_chain_id])
            real_resnum_to_seqres = invert_dict(remapped_sequence_indices_to_seqres[old_chain_id])

            # Compute the seqres resnums for this residue.
            found = False
            fake_resnum = remapped_chain_id[2]
            if fake_resnum in fake_resnum_to_real_resnum:
                real_resnum = fake_resnum_to_real_resnum[fake_resnum]
                if real_resnum in real_resnum_to_seqres:

                    # Map resnum to seqres resnum and get MSA data for this singleton residue.
                    seqres_resnum = real_resnum_to_seqres[real_resnum]
                    msa_tensor = curr_chain_msa_data.msa_to_torch_tensor()

                    # Seqres_resnums are 1-indexed so subtract 1 for index in MSA tensor
                    all_seqres_resnums.append(seqres_resnum)
                    chain_singleton_msa_data.append(msa_tensor[seqres_resnum - 1])
                    found = True

            if not found:
                all_seqres_resnums.append(None)
                chain_singleton_msa_data.append(torch.zeros(21))

        chain.set_seqres_resnums(all_seqres_resnums)
        chain.set_multiple_sequence_alignment_from_singletons(chain_singleton_msa_data, all_seqres_resnums)
        return

    ### Find the old chain ID.
    old_chain_id = all_seg_chid_map[chain.remapped_chain_id]
    _, original_chid = old_chain_id
    chain.set_original_chain_id(old_chain_id)

    ### Handle duplicate chain IDs by splitting on '-' to get original chain ID (a property of mmCIF biological assembly dataset).
    if '-' in original_chid and not original_chid in [x.chid for x in polymer_lig_dict['polymers']]:
        original_chid = original_chid.split('-')[0]

    # Find polymers (crystallized amino acid sequences) parsed from the mmCIF header.
    polymer_seqs = [x for x in polymer_lig_dict['polymers'] if x.chid == original_chid]
    if len(polymer_seqs) == 0:
        return

    # Map from sequences to MSA data to find the MSA data for this chain.
    seq_to_msa_data = {}
    if not msa_data is None:
        seq_to_msa_data = {x.get_query_sequence(): x for x in msa_data.values()}

    # Store the polymer sequence in the chain object.
    polymer_seq = polymer_seqs[0].sequence
    chain.set_polymer_seq(polymer_seq)

    if not polymer_seq in seq_to_msa_data:
        # print("Missing MSA data for chain:", original_chid)
        return 

    curr_chain_msa_data = seq_to_msa_data[polymer_seq]

    ### Remap the residue indices to the crystalized sequence indices.
    fake_resnum_to_real_resnum = invert_dict(residue_reindexing_map[old_chain_id])
    real_resnum_to_seqres = invert_dict(remapped_sequence_indices_to_seqres[old_chain_id])
    seqres_resnums = [real_resnum_to_seqres[fake_resnum_to_real_resnum[x]] for x in chain.resnums]
    chain.set_seqres_resnums(seqres_resnums)
    chain.set_polymer_seq(polymer_seq)

    # Set the MSA data in the chain object.
    # Use the (potentially modified to be the correct key for msa_data) original chid to get MSA data for this chain.
    chain.set_multiple_sequence_alignment(curr_chain_msa_data.msa_to_torch_tensor())
    

def process_row(
        row: pd.Series,
        msa_data_paths: Union[dict, None],
        params: dict
) -> None:
    # Unpickle metadata objects.
    row.pickled_header = pickle.loads(row.pickled_header)
    row.pickled_seg_chain_map_dict = pickle.loads(row.pickled_seg_chain_map_dict)

    # Load MSA data if available.
    msa_data = None
    if msa_data_paths is not None:
        msa_data = load_msa_from_disk(msa_data_paths)
        if msa_data is not None:
            msa_data = {x: y for x,y in msa_data.items()}

    # Extract useful reindexing dictionaries from metadata file.
    residue_reindexing_map = row.pickled_seg_chain_map_dict['residue_reindexing_map']
    all_seg_chid_map = {y:x for x,y in row.pickled_seg_chain_map_dict['seg_chain_map'].items()}

    # Load the protein as a HierView object with remapped chains to iter by (segment, chain).
    try:
        protein_hier_view = load_prody_hierview(row.reduce_output_path) 
    except pr.proteins.pdbfile.PDBParseError:
        print("Couldn't parse PDB file:", row.reduce_output_path)
        return

    # TODO: map polymer sequences to chain data, find MSA data.
    polymer_lig_dict, remapped_sequence_indices_to_seqres = row.pickled_header
    remapped_sequence_indices_to_seqres = reformat_resnum_mapping_dict(remapped_sequence_indices_to_seqres)

    # Loop over the chains and parse the relevant data.
    protein_data = parse_pdb(protein_hier_view, pdb_code=row.reduce_output_path.rsplit('/', 1)[-1].replace('.pdb', ''))
    for chain in protein_data.all_chains:

        # Compute the chi angles for each residue
        chi_angles = compute_chi_angles(chain.residue_coords, chain.sequence_indices)

        # Set the chi angles in the chain object.
        chain.set_chi_angles(chi_angles)
        chain.get_backbone_coords()

        # Remap the chain ID to the original chain ID, and figure out the mapping from residue indexing to MSA data.
        remap_chains(chain, polymer_lig_dict, remapped_sequence_indices_to_seqres, residue_reindexing_map, all_seg_chid_map, msa_data)
    
    output_dict = protein_data.generate_final_data_dict_for_serialization()
    if len(output_dict) > 0:
        output_path = os.path.join(params['reparse_output_path'], '/'.join(row.reduce_output_path.rsplit('/', 2)[-2:]).replace('.pdb', '.pt'))
        torch.save(output_dict, output_path)


def worker_process_row(args):
    torch.set_num_threads(2)
    row, msa_data_paths, params = args
    process_row(row, msa_data_paths, params)


def main(params: dict) -> None:
    """
    Processes all well-formatted proteins using the metadata file produced by reduce_metadata.pkl

    Args:
        params (dict): Dictionary containing parameters.
    """
    metadata = load_metadata(params)
    msa_dataset = MultipleSequenceAlignmentDataset(params)

    # metadata = metadata[metadata.reduce_output_path == '/scratch/bfry/reduce_filtered_pdb_bioasmb/ou/1out_1.pdb']
    # metadata = metadata[metadata.reduce_output_path == '/scratch/bfry/reduce_filtered_pdb_bioasmb/rr/3rri_1.pdb'] # Selenomethionine.
    # metadata = metadata[metadata.reduce_output_path == '/scratch/bfry/reduce_filtered_pdb_bioasmb/ok/5ok7_1.pdb']
    # metadata = metadata[metadata.reduce_output_path == '/scratch/bfry/reduce_filtered_pdb_bioasmb/be/4bei_4.pdb'] # Partially resolved residues.
    # metadata = metadata[metadata.reduce_output_path == '/scratch/bfry/reduce_filtered_pdb_bioasmb/ej/1ej0_1.pdb'] # Singleton chain that looks like AAs.
    # metadata = metadata[metadata.reduce_output_path == '/scratch/bfry/reduce_filtered_pdb_bioasmb/nj/4njb_1.pdb'] 
    # metadata = metadata[metadata.reduce_output_path == '/scratch/bfry/reduce_filtered_pdb_bioasmb/jz/5jzl_1.pdb'] 
    # metadata = metadata[metadata.reduce_output_path == '/scratch/bfry/reduce_filtered_pdb_bioasmb/lu/3lum_1.pdb'] 
    # metadata = metadata[metadata.reduce_output_path == '/scratch/bfry/reduce_filtered_pdb_bioasmb/i5/4i51_1.pdb'] 
    # metadata = metadata[metadata.reduce_output_path == '/scratch/bfry/reduce_filtered_pdb_bioasmb/c5/6c50_1.pdb'] 
    metadata = metadata[metadata.reduce_was_successful == True]
    metadata = metadata.sort_values('reduce_output_path')

    all_args = []
    for _, row in metadata.iterrows():
        pdb_code = row.reduce_output_path.rsplit('/', 1)[-1].replace('.pdb', '').split('_')[0]
        msa_data_paths = msa_dataset.get_pdb_msas(pdb_code)
        all_args.append((row, msa_data_paths, params))


    with multiprocessing.Pool(params['num_workers']) as p:
        for _ in tqdm(p.imap(worker_process_row, all_args), total=len(all_args)):
            pass


    # for _, row in tqdm(metadata.iterrows(), total=len(metadata)):
    #     print(row.reduce_output_path)
    #     pdb_code = row.reduce_output_path.rsplit('/', 1)[-1].replace('.pdb', '').split('_')[0]
    #     msa_data_paths = msa_dataset.get_pdb_msas(pdb_code)
    #     process_row(row, msa_data_paths, params)

    raise NotImplementedError


if __name__ == "__main__":
    import numpy as np
    np.random.seed(1)
    params = {
        'path_to_metadata': './reduce_metadata.pkl',
        'path_to_msa_data': '/nfs/polizzi/ktan/openfold_msa_npys/',
        'path_to_duplicate_chain_file': '/nfs/polizzi/bfry/reparse_pdb/duplicate_pdb_chains.txt',
        'reparse_output_path': '/scratch/bfry/torch_bioasmb_dataset/',
        'num_workers': 20,
    }
    initialize_output_directory(params['reparse_output_path'], '/scratch/bfry/filtered_pdb_bioasmb/')
    main(params)