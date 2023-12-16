import os
from pathlib import Path
from multiprocessing import Pool
from prody import *
import pickle

target_folder = '/nfs/polizzi/cyao/RL/CS184/Project/finals/input_noxaa/ground_truth/'  # CHANGE HERE

def get_pdb_L(p):
    L = len(p.select('name CA').getResnums())
    return L

def process_proteins(basename):
    basename_index = basename.split('_')[-1]
    path_to_target = f'{target_folder}{basename}'
    p = parsePDB(path_to_target)
    L = get_pdb_L(p)
    print(basename_index, L)
    return basename_index, L

def main():
    basenames = [file.name for file in Path(target_folder).glob('*.pdb')]

    # Determine the number of available CPU cores
    num_cores = 4

    # Use multiprocessing to parallelize the processing with all available CPU cores
    with Pool(processes=num_cores) as pool:
        results = pool.map(process_proteins, basenames)

    # Extract results into a dictionary
    length_dict = dict(results)

    # Save the dictionary to a pickle file
    output_pickle_path = '/nfs/polizzi/cyao/RL/CS184/Project/util/lengths_finals.pkl'  # CHANGE HERE
    with open(output_pickle_path, 'wb') as f:
        pickle.dump(length_dict, f)

if __name__ == "__main__":
    main()
