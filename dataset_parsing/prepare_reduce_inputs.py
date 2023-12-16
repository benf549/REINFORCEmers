#!/usr/bin/env python
"""
Reads all mmCIF.gz files in local mirror of RCSB PDB biological assemblies.
Converts those that pass filters to PDB files for input to REDUCE.
See README.md for more details.

(Takes around 15-25 mins to run on np-cpu-1 server with num_parallel_workers=35)
Benjamin Fry (bfry@g.harvard.edu)
"""

import io
import os
import gzip
import shutil
import pickle
import prody as pr
import pandas as pd
from tqdm import tqdm
import multiprocessing
from datetime import date
from typing import List, Dict, Tuple
from utils.parse_mmCIF_header import parse_mmcif_metadata 

# Reduce seems to crash when chains/segments have more than a single letter name.
alphabet = [x for x in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ']

def initialize_output_directory(path_to_pdb_bioasmb_output_dir: str, path_to_mmcif_dataset: str) -> None:
    '''
    Make new directories for the output biological assemblies.
    '''
    if os.path.exists(path_to_pdb_bioasmb_output_dir):
        # Make sure we want to delete the previous output directory.
        _ = input("Are you sure you want to delete the previous output directory? (enter any key to continue, ctrl-c to exit)")
        shutil.rmtree(path_to_pdb_bioasmb_output_dir)

    os.mkdir(path_to_pdb_bioasmb_output_dir)
    for i in os.listdir(path_to_mmcif_dataset):
        os.mkdir(os.path.join(path_to_pdb_bioasmb_output_dir, i))


def get_all_paths(path_to_mmcif_dataset: str) -> List[str]:
    '''
    Finds all paths to .cif.gz files in the biological assembly mmCIF dataset.
    Returns a list of paths to mmCIF.gz files containing biological assemblies.
    '''
    all_paths_list = []
    for subdir in os.listdir(path_to_mmcif_dataset):
        path_to_subdir = os.path.join(path_to_mmcif_dataset, subdir)
        if os.path.isdir(path_to_subdir):
            for cif_file in os.listdir(path_to_subdir):
                cif_path = os.path.join(path_to_subdir, cif_file)
                if cif_path.endswith('.cif.gz'):
                    all_paths_list.append(cif_path)
    return all_paths_list


def load_additional_metadata(path_to_extra_metadata: str) -> Dict[str, Tuple[str, str, str]]:
    '''
    Reads metadata entries file and maps PDB ID to (deposition date, resolution, experiment type).
    returns a dictionary mapping PDB ID to (deposition date, resolution, experiment type).
    '''
    output = {}
    with open(path_to_extra_metadata, 'r') as f:
        for idx, line in enumerate(f.readlines()):
            # Split as TSV.
            data = line.strip().split('\t')

            # Assuming all PDB codes start with numeric character record all of those lines.
            if line[0].isnumeric():
                output[data[0]] = (data[2], data[6], data[7])
    return output


def parse_identifier_from_path(path_to_cif_gz_file: str) -> Tuple[str, str]:
    '''
    Parses the PDB ID from a path to a .cif.gz file.
    returns a tuple of (PDB ID, assembly number).
    '''
    pdb_id, assembly = path_to_cif_gz_file.split('/')[-1].split('.')[0].split('-assembly')
    return pdb_id, assembly


def load_protein(path_to_cif_gz_file: str, pdbid: str) -> Tuple[pr.AtomGroup, Tuple[dict, dict]]:
    '''
    Loads a pdb file from a path and parses header from a .cif.gz file with a single read from disk.
    returns a tuple of (prody protein, header data dictionary).
    '''
    # Push gzip content into string stream and parse.
    output = None

    # decompress and read the gzip file.
    s = io.StringIO()
    with gzip.open(path_to_cif_gz_file, 'rb') as f:
        for line in f.readlines():
            s.write(line.decode('utf-8'))
    s.seek(0)

    # The default prody header parsing doesn't seem to work for any of these files.
    output = pr.parseMMCIFStream(s, header=False)
    header = parse_mmcif_metadata(s, pdbid)

    return output, header


def datestring_to_datetime(datestring: str) -> date:
    '''
    Converts a string representing a date in the format MM/DD/YY to a datetime object.
    Assumes that years 00-29 are 2000-2029 and years 30-99 are 1930-1999.

    Returns a datetime object representing the date.
    '''
    month, day, year = [int(x) for x in datestring.split('/')]

    if str(year)[0] in ['0', '1', '2']:
        year = 2000 + year
    else:
        year = 1900 + year

    return date(year, month, day)


def filter_all_paths(all_paths_list: List[str], extra_metadata: dict, resolution_cutoff: float, date_cutoff: str) -> List[str]:
    '''
    Drops paths that are not a high enough resolution, too new, or not from the right experiment type.
    Returns a list of paths to mmCIF.gz files containing biological assemblies that passed filtering
    '''
    cutoff_date = datestring_to_datetime(date_cutoff)

    output = []
    for path in all_paths_list:
        pdbid, assembly = parse_identifier_from_path(path)
        if pdbid.upper() in extra_metadata:
            pdb_date, resolution, experiment_type = extra_metadata[pdbid.upper()]
            try:
                if float(resolution) < resolution_cutoff and experiment_type in ['X-RAY DIFFRACTION', 'ELECTRON MICROSCOPY'] and datestring_to_datetime(pdb_date) < cutoff_date:
                    output.append(path)
            except:
                pass

    return output


def write_bioasmb_pdb(output_path: str, prody_protein: pr.AtomGroup) -> dict:
    # Get view we can use to iterate segments and chains.
    protein_hierview = prody_protein.getHierView()

    # Store unique segments and chains.
    all_seg_chain_list = []
    for idx, chain in enumerate(protein_hierview):
        # Maps from new segment to (old segment, old chain)
        all_seg_chain_list.append((chain.getSegname(), chain.getChid(), set(chain.getResnums())))

    # Track which segments and chains we want to keep their own chains and which we want to merge.
    to_keep = []
    to_merge = []
    for oldseg, oldchid, resnumset in all_seg_chain_list:
        # Merge residue singleton chains/segments into a single segment/chain.
        if len(resnumset) == 1:
            to_merge.append((oldseg, oldchid, resnumset.pop()))
        else:
            to_keep.append((oldseg, oldchid))

    # Maps from new segment to new chain with alphabet index.
    final_map = {}
    for idx, pair in enumerate(to_keep):
        curr_seg = alphabet[idx // len(alphabet)]
        curr_chid = alphabet[idx % len(alphabet)]
        final_map[pair] = (curr_seg, curr_chid)

    # Maps the old segment and chain to new segment, chain, and resnum for singleton residues we are merging.
    remap_index = len(to_keep)
    singleton_map = {}
    for jdx, pair in enumerate(to_merge):
        # Don't need old resnum because it was a singleton.
        oldseg, oldchid, oldresnum = pair
        pair = (oldseg, oldchid)
        curr_seg = alphabet[remap_index // len(alphabet)]
        curr_chid = alphabet[remap_index % len(alphabet)]
        singleton_map[pair] = (curr_seg, curr_chid, jdx + 1)

    residue_reindexing_map = {}
    # Update the segment and chain names.
    for chain in protein_hierview:
        # Get the old segment and chain.
        old_seg, old_chid = chain.getSegname(), chain.getChid()

        if (old_seg, old_chid) in final_map:
            # Get the new segment and chain.
            new_seg, new_chid = final_map[(old_seg, old_chid)]

            # Update the segment and chain.
            chain.setSegnames(new_seg)
            chain.setChid(new_chid)

            # Record the residue reindexing map.
            resnum_remap = {y:x+1 for x,y in enumerate(sorted(list(set(chain.getResnums()))))}
            chain.setResnums([resnum_remap[x] for x in chain.getResnums()])
            residue_reindexing_map[(old_seg, old_chid)] = resnum_remap
        else:
            # For singleton residues, get the new segment, chain, and resnum.
            new_seg, new_chid, new_resnum = singleton_map[(old_seg, old_chid)]

            chain.setSegnames(new_seg)
            chain.setChid(new_chid)

            resnum_remap = {y: new_resnum for y in sorted(list(set(chain.getResnums())))}
            chain.setResnums([resnum_remap[x] for x in chain.getResnums()])
            residue_reindexing_map[(old_seg, old_chid)] = resnum_remap
        

    # Write the output to the output directory.
    pr.writePDB(output_path, prody_protein)

    # Merge the singleton residues into the final map. Merge the residue reindexing maps.
    final_map = {**final_map, **singleton_map}
    final_map = {'residue_reindexing_map': residue_reindexing_map, 'seg_chain_map': final_map}

    return final_map

def worker_wrapper(args):
    """
    Extracts header metadata and copies PDB file to PDB format from mmCIF.gz format only if file contains protein residues and is not too long.
    Returns tuple of (path, was_parsed, length, pickled_header, output_path, pickled_seg_chain_map_dict)
    """

    # Arguments:
    path, max_bioasm_len, path_to_pdb_bioasmb_output_dir = args

    # Read the pdb from mmCIF.
    pdbid, assembly = parse_identifier_from_path(path)

    try:
        # Attempt to load the PDB file by parsing header and structure.
        prody_protein, header = load_protein(path, pdbid.upper())
    except ValueError:
        print('FAILED:', path)
        return path, False, None, None, None, None

    try:
        # Filter out anything too long or empty
        if len(prody_protein.protein.ca) > max_bioasm_len: # type: ignore
            return path, False, len(prody_protein.protein.ca), None, None, None # type: ignore
    except AttributeError:
        # Handle no protein residues in structure (by ignoring these).
        return path, False,  0, None, None, None

    # Write the output to the output directory.
    output_path = os.path.join(path_to_pdb_bioasmb_output_dir, pdbid[1:3], f'{pdbid}_{assembly}.pdb')
    try:
        seg_chain_map = write_bioasmb_pdb(output_path, prody_protein)
    except IndexError:
        # This seems to only happen for some (3) weird antibiotic peptide bundles so doesn't seem to be worth handling.
        print('FAILED (too many chains):', path)
        return path, False, len(prody_protein), None, None, None

    return path, True, len(prody_protein.protein.ca), pickle.dumps(header), output_path, pickle.dumps(seg_chain_map) # type: ignore


def main(
        path_to_pdb_bioasmb_output_dir: str, 
        path_to_mmcif_dataset: str, 
        path_to_extra_metadata: str, 
        max_bioasm_len: int, 
        resolution_cutoff: float, 
        date_cutoff: str, 
        num_parallel_workers: int, 
        output_metadata_path: str
) -> None:

    # Prepare output directories
    initialize_output_directory(path_to_pdb_bioasmb_output_dir, path_to_mmcif_dataset)

    # Get all paths to .cif.gz files in the biological assembly mmCIF dataset.
    all_paths = get_all_paths(path_to_mmcif_dataset)

    # Get dictionary mapping from PDB ID to (deposition date, resolution, experiment type).
    extra_metadata = load_additional_metadata(path_to_extra_metadata)

    # Don't parse things that are not high enough resolution, too new, or not right experiment type.
    all_paths = filter_all_paths(all_paths, extra_metadata, resolution_cutoff, date_cutoff)

    # Write all bioassemblies that passed filtering to output directory.
    metadata = {}
    with multiprocessing.Pool(num_parallel_workers) as p:
        for output in tqdm(p.imap(worker_wrapper, [(path, max_bioasm_len, path_to_pdb_bioasmb_output_dir) for path in all_paths]), total=len(all_paths)):
            path, was_parsed, length, pickled_header, output_path, pickled_seg_chain_map_dict = output
            metadata[path] = (was_parsed, length, output_path, pickled_header, pickled_seg_chain_map_dict)

    # Construct a dataframe from all the metadata we parsed and save it to working directory.
    df = pd.DataFrame.from_dict(metadata, orient='index')
    df.columns = ['was_parsed', 'bioasmb_length', 'filtering_output_path', 'pickled_header', 'pickled_seg_chain_map_dict']
    df.to_pickle(output_metadata_path)


if __name__ == "__main__":
    path_to_mmcif_bioasmb_dataset = '/nfs/polizzi/shared/databases/mmCIF_bioasmb_pdb_mirror/mmCIF_bioasmb'
    path_to_extra_metadata = '/nfs/polizzi/shared/databases/mmCIF_bioasmb_pdb_mirror/entries.idx'
    path_to_pdb_bioasmb_output_dir = '/scratch/bfry/filtered_pdb_bioasmb_/'
    output_metadata_path = './filtering_metadata_.pkl'
    num_parallel_workers = 25

    max_bioasm_len = 10_000
    resolution_cutoff = 3.5
    date_cutoff = "06/01/22"

    main(path_to_pdb_bioasmb_output_dir, path_to_mmcif_bioasmb_dataset, path_to_extra_metadata, max_bioasm_len, resolution_cutoff, date_cutoff, num_parallel_workers, output_metadata_path)
