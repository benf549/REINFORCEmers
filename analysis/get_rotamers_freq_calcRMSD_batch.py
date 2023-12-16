import pandas as pd
import os
import argparse
import pickle
import numpy as np
from prody import *
import combs2
from multiprocessing import Pool
from pathlib import Path

## =====  CHANGE HERE  ======== 
target_folder = '/nfs/polizzi/cyao/RL/CS184/Project/all_test_supervised_repacked_best_supervised_noxaa/ground_truth/'
mobile_folder = '/nfs/polizzi/cyao/RL/CS184/Project/all_test_supervised_repacked_best_supervised_noxaa/repacked/'
output_folder = '/nfs/polizzi/cyao/RL/CS184/Project/all_test_supervised_repacked_best_supervised_noxaa/1_rmsd_rotafreq/' # output
template_folder = '/nfs/polizzi/cyao/RL/CS184/Project/all_test_supervised_repacked_best_supervised_noxaa/0_template/' # output
os.system(f'mkdir -p {output_folder}')
os.system(f'mkdir -p {template_folder}')
    
    
    
    
def parsing_line(line):
    if 'SUMMARY' not in line and "residue:occupancy" not in line:
        parts = line.split()
        first_five = line[1:6]
        residue = first_five[0]
        position = first_five[1:].strip()
        residue_name = parts[-1].split(':')[0]
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

        try: ## too many errors... :(
            if row.resname not in ['GLY', 'ALA', 'PRO']:
                bbdep_freqs.append(round(rota_dict_tmp[ABPLE][row.rotamer], 6))
                bbind_freqs.append(round(rota_dict_tmp['bb_ind'][row.rotamer], 6))
                
            elif row.resname in ['PRO'] and ABPLE in ['A','B','P']:
                bbdep_freqs.append(round(rota_dict_tmp[ABPLE][row.rotamer], 6))
                bbind_freqs.append(round(rota_dict_tmp['bb_ind'][row.rotamer], 6))

                
            else:
                bbind_freqs.append(np.nan)
                bbdep_freqs.append(np.nan)
                
        except:
            bbind_freqs.append(np.nan)
            bbdep_freqs.append(np.nan)

    df_rota['bbind_freq'] = bbind_freqs
    df_rota['bbdep_freq'] = bbdep_freqs
    return df_rota




def process_proteins(target_file):

    target_path = os.path.join(target_folder, target_file)
    mobile_path = os.path.join(mobile_folder, target_file)
    output_path = os.path.join(output_folder, target_file.replace('.pdb', '.pkl'))
    template_path = os.path.join(template_folder, target_file.replace('.pdb', '.pkl'))

    # Run your analysis functions
    ## load rotamer frequency dict
    path_to_combs_rotamer_files = '/nfs/polizzi/cyao/Combs2/combs2/files/rotamer_freqs.pkl'
    with open(path_to_combs_rotamer_files, 'rb') as f:
        rota_freq_dict = pickle.load(f)
    df_combined = analyze_proteins(target_path, mobile_path, output_path, template_path, rota_freq_dict, target_file)

    # Save the result to pickle
    df_combined.to_pickle(output_path)
    print(f'Processed {target_file} and saved results to {output_path}')


def analyze_proteins(target_path, mobile_path, output_path, template_path, rota_freq_dict,target_file):
    # ... (the content of your analysis functions)
    path_to_output_folder = '/'.join(output_path.split('/')[0:-1])
    tmp_txt = target_file.replace('.pdb', '_tmp.txt')
    path_to_output_text = f'{path_to_output_folder}/{tmp_txt}'
    ## ====== run rotalyze ========
    df_rota = run_rotalyze(mobile_path, path_to_output_text)

    
    ## ====== analyze rotamer frequency ========
    df_template = run_template(template_path, mobile_path)
    df_rota = add_rota_freq(df_rota, df_template, rota_freq_dict)
    
    
    ## ====== get rmsd =======
    protein1, protein2 = parsePDB(target_path), parsePDB(mobile_path)
    df_rmsd_sc = calculate_RMSDs_prody(protein1, protein2)
    df_rmsd_sc['resnum'] = df_rmsd_sc['resnum'].astype(str)


    ## ====== combine rotamer frequency to rmsd =======
    combined_df = pd.merge(df_rmsd_sc, df_rota, on=['resnum','resname'], how='outer')
    basename = target_path.split('/')[-1]
    combined_df['basename'] = [basename]*len(combined_df)


    return combined_df


def main():
    
    # List all target files in the target folder
    target_files = [file.name for file in Path(target_folder).glob('*.pdb')]

    # Determine the number of available CPU cores
    num_cores = 64

    # Use multiprocessing to parallelize the processing with all available CPU cores
    with Pool(processes=num_cores) as pool:
        pool.map(process_proteins, target_files)

if __name__ == "__main__":
    main()

