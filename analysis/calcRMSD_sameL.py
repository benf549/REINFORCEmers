import os
from prody import *
import argparse
from multiprocessing import Pool




def calculate_RMSDs(path_to_protein1, path_to_protein2):
    # Load your PDB files or fetch them from an online database
    # protein1 is target, protein2 is mobile, resnums_to_align is a list of resnums for alignment
    protein1 = parsePDB(path_to_protein1)  
    protein2 = parsePDB(path_to_protein2)  

    ## get resnums for alignment
    resnums_to_align_1 = list(set(protein1.select('name CA').getResnums()))
    resnums_to_align_2 = list(set(protein2.select('name CA').getResnums())) ## this assumes that they both are  in the same length
    
    assert len(resnums_to_align_1) == len(resnums_to_align_2), "Proteins are not of the same length."

    
    # Select Calpha atoms for the specified residue numbers for alignment
    
    ca_protein1 = protein1.select('backbone and not element H and resnum ' + ' '.join(map(str, resnums_to_align_1)))
    ca_protein2 = protein2.select('backbone and not element H and resnum ' + ' '.join(map(str, resnums_to_align_2)))

    # Calculate and apply the transformation matrix 
    try:
        calcTransformation(ca_protein2, ca_protein1).apply(protein2)
    except:
        print(f'========= This protein is probably not in the same length: {path_to_protein1} & {path_to_protein2} ================')


    # Calculate main chain RMSD between aligned proteins
    rmsd_mc = calcRMSD(ca_protein2, ca_protein1) # the rmsd of the main chain atoms


    ## Calculate side chain RMSD (before and individually placing 
    for resnum_1, resnum_2 in zip(resnums_to_align_1, resnums_to_align_2):
        residue_1 = protein1.select(f'resnum {resnum_1}')
        residue_2 = protein2.select(f'resnum {resnum_2}')
        
        residue_1_mc = residue_1.select('backbone')
        residue_1_sc = residue_1.select('sidechain')
        
        residue_2_mc = residue_2.select('backbone')
        residue_2_sc = residue_2.select('sidechain')
        
        ## side chain RMSD (crude)
        rmsd_sc_crude = calcRMSD(residue_1_sc, residue_2_sc)

        ## side chain RMSD (refined)
        calcTransformation(residue_2_mc, residue_1_mc).apply(residue_2)
        residue_2_sc = residue_2.select('sidechain') # here residue_2 is after superposition
        rmsd_sc_refined = calcRMSD(residue_1_sc, residue_2_sc)
        
        print(rmsd_mc, rmsd_sc_crude, rmsd_sc_refined)
        
    
    return rmsd_mc, rmsd_sc_crude, rmsd_sc_refined
        



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="super pose protein with the same length to a target")
    parser.add_argument('--path_to_target', '-t', type=str, help="Path to the folder that contains all pdbs of interest", required=True)
    parser.add_argument('--path_to_mobile', '-m', type=str, help="Path to the folder that contains all pdbs of interest", required=True)
    args = parser.parse_args()    
    
    path_to_target = args.path_to_target
    path_to_mobile = args.path_to_mobile
    
    calculate_RMSDs(path_to_target, path_to_mobile)
    
    










