#!/usr/bin/env python3
"""
Executes REDUCE on our filtered PDB dataset in parallel.
See README.md for more details.

Takes around 5 hours with 35 processes.
Benjamin Fry (bfry@g.harvard.edu)
"""

import multiprocessing
import subprocess
import pandas as pd
import os
from tqdm import tqdm

from prepare_reduce_inputs import initialize_output_directory

def run_reduce_on_file(args):
    """
    Run reduce on a single PDB file containing a biological assembly and any heteroatoms.
    """
    path_to_pdb_file, path_to_reduce, path_to_reduce_output_dir, path_to_hetatom_dict = args

    path_prefix = path_to_pdb_file.rsplit('/', 2)[-2:]
    output_path = os.path.join(path_to_reduce_output_dir, *path_prefix)
    temp_output_path = output_path + '.temp'

    try:
        p1 = subprocess.run(f'{path_to_reduce} -TRIM {path_to_pdb_file} -Quiet > {temp_output_path}', shell=True, capture_output=True)
        p2 = subprocess.run(f'{path_to_reduce} -DB {path_to_hetatom_dict} -BUILD {temp_output_path} -Quiet > {output_path}', shell=True, capture_output=True)
        os.remove(temp_output_path)

        if p1.returncode in [0, 255] and p2.returncode in [0, 1]:
            return path_to_pdb_file, True, output_path
        else:
            print("FAILED: ", path_to_pdb_file, p1.returncode, p2.returncode)
            # print(p2.stderr.decode('utf-8'))
            return path_to_pdb_file, False, None
    except Exception as e:
        print("FAILED: ", path_to_pdb_file)
        # print(e)
        return path_to_pdb_file, False, None


def main(
    path_to_reduce: str, 
    path_to_pdb_bioasmb_input_dir: str,
    path_to_reduce_output_dir: str, 
    path_to_hetatom_dict: str, 
    path_to_input_metadata: str, 
    path_to_output_metadata: str,
    num_parallel_workers: int, 
) -> None:
    # Create output file structure.
    initialize_output_directory(path_to_reduce_output_dir, path_to_pdb_bioasmb_input_dir)

    # Get files that passed filtering.
    metadata = pd.read_pickle(path_to_input_metadata)
    pdb_file_paths = sorted(metadata[metadata.was_parsed].filtering_output_path.tolist())

    success_results = {}
    reduce_path_results = {}
    with multiprocessing.Pool(num_parallel_workers) as p:
        for output in tqdm(p.imap(run_reduce_on_file, [(path, path_to_reduce, path_to_reduce_output_dir, path_to_hetatom_dict) for path in pdb_file_paths]), total=len(pdb_file_paths)):
            path, was_successful, output_path = output
            success_results[path] = was_successful
            reduce_path_results[path] = output_path
    
    # Add output to metadata dictionary
    metadata['reduce_was_successful'] = metadata.filtering_output_path.map(success_results)
    metadata['reduce_output_path'] = metadata.filtering_output_path.map(reduce_path_results)
    metadata.to_pickle(path_to_output_metadata)


if __name__ == "__main__":
    params = {
        'path_to_reduce': '/nfs/polizzi/bfry/programs/reduce/reduce',
        'path_to_hetatom_dict': '/nfs/polizzi/bfry/programs/reduce/reduce_wwPDB_het_dict.txt',
        'path_to_pdb_bioasmb_input_dir': '/scratch/bfry/filtered_pdb_bioasmb_/',
        'path_to_reduce_output_dir': '/scratch/bfry/reduce_filtered_pdb_bioasmb_/',
        'path_to_input_metadata': './filtering_metadata_.pkl',
        'path_to_output_metadata': './reduce_metadata_.pkl',
        'num_parallel_workers': 25,
    }
    main(**params)
