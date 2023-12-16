from prody import *
import os
from multiprocessing import Pool

def process_pdb(pdb_file, path_to_protein, path_to_protein_noxaa):
    if '.pdb' in pdb_file:
        p = parsePDB(f'{path_to_protein}{pdb_file}')

        try:
            xaa = p.select('resname XAA')
            xaa.setResnames('GLY')
            p_new = p.select('not resname XAA') + xaa
            writePDB(f'{path_to_protein_noxaa}{pdb_file}', p_new)
        except:
            writePDB(f'{path_to_protein_noxaa}{pdb_file}', p)


if __name__ == '__main__':
    for subfolder in ['supervised','ground_truth','reinforce']:
        path_to_protein = f'/nfs/polizzi/cyao/RL/CS184/Project/finals/input/{subfolder}/' # input
        path_to_protein_noxaa = f'/nfs/polizzi/cyao/RL/CS184/Project/finals/input_noxaa/{subfolder}/' # output 
        os.system(f'mkdir -p {path_to_protein_noxaa}')

        pdb_files = [file for file in os.listdir(path_to_protein) if '.pdb' in file]

        # Set the number of processes to the number of available CPU cores or adjust as needed
        num_processes = 32

        with Pool(num_processes) as pool:
            pool.starmap(process_pdb, [(pdb_file, path_to_protein, path_to_protein_noxaa) for pdb_file in pdb_files])

    