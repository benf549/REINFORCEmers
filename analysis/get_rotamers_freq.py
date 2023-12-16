import pandas as pd
import os
import argparse
import pickle
import numpy as np

def parsing_line(line):
    if 'SUMMARY' not in line and "residue:occupancy" not in line:
        parts = line.split()
        residue = parts[0]
        position = parts[1]
        residue_name = parts[2].split(':')[0]
        if 'OUTLIER' in line:
            rotamer = 'OUTLIER'
        else:
            rotamer = parts[-1].split(':')[-1]
        data_line = [residue, position, residue_name, rotamer]
        return data_line


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process input files and store the results in a pickle file.")
    parser.add_argument('--path_to_mobile', '-m', type=str, help="Path to the folder containing the input files")
    parser.add_argument("--path_to_output_pickle", '-o', help="Path to the output pickle file")
    args = parser.parse_args()

    path_to_mobile = args.path_to_mobile
    path_to_output_pickle = args.path_to_output_pickle

    path_to_output_folder = '/'.join(path_to_output_pickle.split('/')[0:-1])
    path_to_output_text = f'{path_to_output_folder}tmp.txt'
    os.system(f'mkdir -p {path_to_output_folder}')

    # Step 1: Run phenix.rotalyze
    # print(path_to_mobile)
    os.system(f'phenix.rotalyze model={path_to_mobile} > {path_to_output_text}')

    # Step 2: Parse phenix result
    with open(path_to_output_text, 'r') as file:
        data = file.readlines()

    parsed_data = []
    for line in data:
        data_line = parsing_line(line)
        if data_line is not None:
            parsed_data.append(data_line)

    columns = ["chain", "resnum", "resname", "rotamer"]

    # Create a DataFrame
    df = pd.DataFrame(parsed_data, columns=columns)

    # Add backbone-independent rotamer frequency
    path_to_combs_rotamer_files = '/nfs/polizzi/cyao/Combs2/combs2/files/rotamer_freqs.pkl'
    with open(path_to_combs_rotamer_files, 'rb') as f:
        rota_freq_dict = pickle.load(f)

    bbind_freqs = []
    for _, row in df.iterrows():
        resname = row.resname
        rotamer = row.rotamer

        if resname in ['GLY', 'ALA']:
            rotamer = np.nan
            bbind_freq = np.nan
        else:
            rota_dict_tmp = rota_freq_dict[resname]
            bbind_freq = round(rota_dict_tmp['bb_ind'][rotamer], 6)
        bbind_freqs.append(bbind_freq)

    df['bbind_freq'] = bbind_freqs
    # print(df)

    # Save DataFrame to pickle file
    with open(path_to_output_pickle, 'wb') as f:
        pickle.dump(df, f)

    print(f'====== Pickle file of the DataFrame saved in {path_to_output_pickle}')



if __name__ == "__main__":
    main()
