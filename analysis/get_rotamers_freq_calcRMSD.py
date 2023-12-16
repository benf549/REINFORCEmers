import pandas as pd
import os
import argparse
import pickle
import numpy as np
from prody import *
import combs2

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
    rmsd_mc_list, rmsd_sc_crude_list, rmsd_sc_refined_list, resnum_list, resname_list, beta_list = [], [], [], [], [], []

    for resnum_1, resnum_2 in zip(resnums_to_align_1, resnums_to_align_2):
        # print(resnum_1, resnum_2)
        residue_1 = protein1.select(f'resnum {resnum_1}')
        residue_2 = protein2.select(f'resnum {resnum_2}')
        beta_2 = residue_2.getBetas()[0]
        
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
        # print(resname, resnum_1, rmsd_mc, rmsd_sc_crude,rmsd_sc_refined)

        resnum_list.append(int(resnum_2))
        resname_list.append(resname)
        rmsd_mc_list.append(rmsd_mc)
        rmsd_sc_crude_list.append(rmsd_sc_crude)
        rmsd_sc_refined_list.append(rmsd_sc_refined)
        beta_list.append(beta_2)

    df_rmsd_sc['resnum'] = resnum_list
    df_rmsd_sc['resname'] = resname_list
    df_rmsd_sc['rmsd_mc'] = rmsd_mc_list
    df_rmsd_sc['rmsd_sc_crude'] = rmsd_sc_crude_list
    df_rmsd_sc['rmsd_sc_refined'] = rmsd_sc_refined_list
    df_rmsd_sc['Bfactor'] = beta_list
    
    return df_rmsd_sc


def run_rotalyze(path_to_mobile, path_to_output_text):
    os.system(f'phenix.rotalyze model={path_to_mobile} > {path_to_output_text}')

    with open(path_to_output_text, 'r') as file:
        data = file.readlines()

    parsed_data = [parsing_line(line) for line in data if parsing_line(line) is not None]
    columns = ["chain", "resnum", "resname", "rotamer"]
    df_rota = pd.DataFrame(parsed_data, columns=columns)
    return df_rota
    
    
def run_template(path_to_template_pickle, path_to_target):
    if not os.path.exists(path_to_template_pickle): # load the template if it's already calculated
        template = combs2.design.template.Template(f'{path_to_target}')
        df_template = template.dataframe
        with open(path_to_template_pickle, 'wb') as f:
            pickle.dump(df_template,f)
    else:
        with open(path_to_template_pickle, 'rb') as f:
            df_template = pickle.load(f)
    # print(df_template)
    return df_template
    
def add_rota_freq(df_rota, df_template, rota_freq_dict):
    bbind_freqs = []
    bbdep_freqs = []

    for _, row in df_rota.iterrows():
        resnum = int(row.resnum)
        # print(resnum)
        df_sub = df_template[df_template.resnum == resnum]
        # print(df_sub)
        ABPLE = df_sub.ABPLE.iloc[0]
        rota_dict_tmp = rota_freq_dict[row.resname]

        if row.resname not in ['GLY', 'ALA']:
            bbind_freqs.append(round(rota_dict_tmp['bb_ind'][row.rotamer], 6))
            bbdep_freqs.append(round(rota_dict_tmp[ABPLE][row.rotamer], 6))
        else:
            bbind_freqs.append(np.nan)
            bbdep_freqs.append(np.nan)

    df_rota['bbind_freq'] = bbind_freqs
    df_rota['bbdep_freq'] = bbdep_freqs
    return df_rota



def main():
    ## ===== input =========
    
    parser = argparse.ArgumentParser(description="Process input files and store the results in a pickle file.")
    parser.add_argument('--path_to_target', '-t', type=str, help="Path to the folder that contains all pdbs of interest", required=True)
    parser.add_argument('--path_to_mobile', '-m', type=str, help="Path to the folder containing the input files")
    parser.add_argument("--path_to_output_pickle", '-o', help="Path to the output pickle file")
    parser.add_argument("--path_to_template_pickle", '-p', help="Path to the output template file")
    args = parser.parse_args()

    path_to_target, path_to_mobile, path_to_output_pickle, path_to_template_pickle = args.path_to_target, args.path_to_mobile, args.path_to_output_pickle, args.path_to_template_pickle
    path_to_output_tsv = path_to_output_pickle.split('.p')[0]+'.tsv'
    path_to_output_folder = '/'.join(path_to_output_pickle.split('/')[0:-1])
    path_to_output_text = f'{path_to_output_folder}tmp.txt'
    basename = path_to_mobile.split('/')[-1]
    os.system(f'mkdir -p {path_to_output_folder}')
    
    ## load rotamer frequency dict
    path_to_combs_rotamer_files = '/nfs/polizzi/cyao/Combs2/combs2/files/rotamer_freqs.pkl'
    with open(path_to_combs_rotamer_files, 'rb') as f:
        rota_freq_dict = pickle.load(f)


    ## ====== run rotalyze ========
    df_rota = run_rotalyze(path_to_mobile, path_to_output_text)

    
    ## ====== analyze rotamer frequency ========
    df_template = run_template(path_to_template_pickle, path_to_target)
    df_rota = add_rota_freq(df_rota, df_template, rota_freq_dict)
    
    
    ## ====== get rmsd =======
    protein1, protein2 = parsePDB(path_to_target), parsePDB(path_to_mobile)
    df_rmsd_sc = calculate_RMSDs_prody(protein1, protein2)
    df_rmsd_sc['resnum'] = df_rmsd_sc['resnum'].astype(str)


    ## ====== combine rotamer frequency to rmsd =======
    combined_df = pd.merge(df_rmsd_sc, df_rota, on=['resnum','resname'], how='outer')
    combined_df['basename'] = [basename]*len(combined_df)

    with open(path_to_output_pickle, 'wb') as f:
        pickle.dump(combined_df, f)

    combined_df.to_csv(path_to_output_tsv, sep='\t')
    

    print(f'====== Pickle file of the DataFrame saved in {path_to_output_pickle} ========')
    print(f'====== tsv file of the DataFrame saved in {path_to_output_tsv} =========')



if __name__ == "__main__":
    main()
