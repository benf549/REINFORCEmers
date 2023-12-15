import pandas as pd
import os
import argparse
import pickle
import numpy as np
from prody import *

def parsing_line(line):
    if 'SUMMARY' not in line and "residue:occupancy" not in line:
        parts = line.split()
        residue = parts[0]
        position = parts[1]
        residue_name = parts[2].split(':')[0]
        rotamer = 'OUTLIER' if 'OUTLIER' in line else parts[-1].split(':')[-1]
        return [residue, position, residue_name, rotamer]

def calculate_RMSDs_prody(protein1, protein2):
    resnums_to_align_1 = list(set(protein1.select('name CA').getResnums()))
    resnums_to_align_2 = list(set(protein2.select('name CA').getResnums()))
        
        
    ca_protein1 = protein1.select('backbone and not element H and resnum ' + ' '.join(map(str, resnums_to_align_1)))
    ca_protein2 = protein2.select('backbone and not element H and resnum ' + ' '.join(map(str, resnums_to_align_2)))

    try:
        calcTransformation(ca_protein2, ca_protein1).apply(protein2)
    except:
        print(f'========= These proteins are probably not in the same length =========')

    rmsd_mc = calcRMSD(ca_protein2, ca_protein1)

    df_rmsd_sc = pd.DataFrame()
    rmsd_mc_list, rmsd_sc_crude_list, rmsd_sc_refined_list, resnum_list, resname_list = [], [], [], [], []

    for resnum_1, resnum_2 in zip(resnums_to_align_1, resnums_to_align_2):
        print(resnum_1, resnum_2)
        residue_1 = protein1.select(f'resnum {resnum_1}')
        residue_2 = protein2.select(f'resnum {resnum_2}')
        
        residue_1_mc, residue_1_sc = residue_1.select('backbone'), residue_1.select('sidechain and not element H')
        residue_2_mc, residue_2_sc = residue_2.select('backbone'), residue_2.select('sidechain and not element H')



        if residue_1_sc == None:
            rmsd_sc_crude = 0
            rmsd_sc_refined = 0
            
        else:
            rmsd_sc_crude = calcRMSD(residue_1_sc, residue_2_sc)

            calcTransformation(residue_2_mc, residue_1_mc).apply(residue_2)
            residue_2_sc = residue_2.select('sidechain')
            rmsd_sc_refined = calcRMSD(residue_1_sc, residue_2_sc)

        resname = residue_2.select('name CA').getResnames()[0]
        print(resname, resnum_1, rmsd_mc, rmsd_sc_crude,rmsd_sc_refined)

        resnum_list.append(int(resnum_2))
        resname_list.append(resname)
        rmsd_mc_list.append(rmsd_mc)
        rmsd_sc_crude_list.append(rmsd_sc_crude)
        rmsd_sc_refined_list.append(rmsd_sc_refined)

    df_rmsd_sc['resnum'] = resnum_list
    df_rmsd_sc['resname'] = resname_list
    df_rmsd_sc['rmsd_mc'] = rmsd_mc_list
    df_rmsd_sc['rmsd_sc_crude'] = rmsd_sc_crude_list
    df_rmsd_sc['rmsd_sc_refined'] = rmsd_sc_refined_list
    
    return df_rmsd_sc



def main():
    ## ===== input =========
    
    parser = argparse.ArgumentParser(description="Process input files and store the results in a pickle file.")
    parser.add_argument('--path_to_target', '-t', type=str, help="Path to the folder that contains all pdbs of interest", required=True)
    parser.add_argument('--path_to_mobile', '-m', type=str, help="Path to the folder containing the input files")
    parser.add_argument("--path_to_output_pickle", '-o', help="Path to the output pickle file")
    args = parser.parse_args()

    path_to_target, path_to_mobile, path_to_output_pickle = args.path_to_target, args.path_to_mobile, args.path_to_output_pickle
    path_to_output_tsv = path_to_output_pickle.split('.pkl')[0]+'.tsv'
    path_to_output_folder = '/'.join(path_to_output_pickle.split('/')[0:-1])
    path_to_output_text = f'{path_to_output_folder}tmp.txt'
    basename = path_to_mobile.split('/')[-1]
    os.system(f'mkdir -p {path_to_output_folder}')


    ## ====== run rotalyze ========
    os.system(f'phenix.rotalyze model={path_to_mobile} > {path_to_output_text}')

    with open(path_to_output_text, 'r') as file:
        data = file.readlines()

    parsed_data = [parsing_line(line) for line in data if parsing_line(line) is not None]
    columns = ["chain", "resnum", "resname", "rotamer"]
    df = pd.DataFrame(parsed_data, columns=columns)

    
    ## ====== analyze rotamer frequency ========
    path_to_combs_rotamer_files = '/nfs/polizzi/cyao/Combs2/combs2/files/rotamer_freqs.pkl'
    with open(path_to_combs_rotamer_files, 'rb') as f:
        rota_freq_dict = pickle.load(f)

    bbind_freqs = [round(rota_freq_dict[row.resname]['bb_ind'][row.rotamer], 6) if row.resname not in ['GLY', 'ALA'] else np.nan for _, row in df.iterrows()]
    df['bbind_freq'] = bbind_freqs
    

    
    ## ====== get rmsd =======
    protein1, protein2 = parsePDB(path_to_target), parsePDB(path_to_mobile)
    print(protein1)
    print(protein2)
    df_rmsd_sc = calculate_RMSDs_prody(protein1, protein2)
    df_rmsd_sc['resnum'] = df_rmsd_sc['resnum'].astype(str)

    ## ====== combine rotamer frequency to rmsd =======
    combined_df = pd.merge(df_rmsd_sc, df, on=['resnum','resname'], how='outer')

    with open(path_to_output_pickle, 'wb') as f:
        pickle.dump(combined_df, f)

    combined_df['basename'] = [basename]*len(combined_df)
    combined_df.to_csv(path_to_output_tsv, sep='\t')
    

    print(f'====== Pickle file of the DataFrame saved in {path_to_output_pickle} ========')
    print(f'====== tsv file of the DataFrame saved in {path_to_output_tsv} =========')

if __name__ == "__main__":
    main()
