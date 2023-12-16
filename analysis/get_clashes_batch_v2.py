import subprocess
import os
import pandas as pd
from pathlib import Path
from multiprocessing import Pool

PROBE_CMD = '/programs/x86_64-linux/probe/2.16.130520/probe -U -CON -Explicit -NOFACE -WEAKH -DE32 -WAT2wat -4 -ON -MC ALL ALL -'
header = 'reinforce' # or reinforce/ supervised
target_folder = '/nfs/polizzi/cyao/RL/CS184/Project/finals/input_noxaa/ground_truth/'
mobile_folder = f'/nfs/polizzi/cyao/RL/CS184/Project/finals/input_noxaa/{header}/'
output_folder = f'/nfs/polizzi/cyao/RL/CS184/Project/finals/2_clashes/{header}/' # output
os.system(f'mkdir -p {output_folder}')



def parse_probe_output(probe_output):
    '''
    Parse the probe output into a pd.DataFrame where each row corresponds to a pair of contacting atoms
    '''
    rows = []
    for line in probe_output.split('\n'):
        try:
            _, _, contact_type, atom1, atom2, *_ = line.split(':')
            chain1   = atom1[:2].strip()
            resnum1  = int(atom1[2:6])
            resname1 = atom1[6:10].strip()
            name1    = atom1[10:15].strip()
            chain2   = atom2[:2].strip()
            resnum2  = int(atom2[2:6])
            resname2 = atom2[6:10].strip()
            name2    = atom2[10:15].strip()
        except ValueError as e:
            # sys.stderr.write(f"failed to parse line: {line}")
            continue

        rows.append(dict(
            chain1=chain1, resnum1=resnum1, resname1=resname1, name1=name1,
            chain2=chain2, resnum2=resnum2, resname2=resname2, name2=name2,
            contact_type=contact_type,
        ))
    return pd.DataFrame(rows)




def get_hb_clashes_df(pdb_file):
    ground_truth = 'ground_truth' in pdb_file
    basename = os.path.basename(pdb_file)
    with open(pdb_file) as f:
        pdb = f.read()
            
    probe_out = subprocess.run(PROBE_CMD.split(), input=pdb, text=True, capture_output=True).stdout

    contacts_df = parse_probe_output(probe_out)

    hb_clashes_df = contacts_df[contacts_df.contact_type.isin(['hb','bo'])]
    
    # Use .loc to set values without the SettingWithCopyWarning
    hb_clashes_df.loc[:, 'basename'] = basename
    if ground_truth:
        hb_clashes_df.loc[:, 'model_type'] = 'ground_truth'
    else:
        hb_clashes_df.loc[:, 'model_type'] = header
    hb_clashes_df.loc[:, 'ground_truth'] = ground_truth
    
    return hb_clashes_df



def process_proteins(basename):
    
    basename_index_pdb = basename.split('_')[-1]
    basename_index = basename_index_pdb.split('.pdb')[0]
    output_path_pkl = f'{output_folder}{header}_{basename_index}.pkl'
    print(output_path_pkl)
    
    path_to_target = f'{target_folder}{basename}'
    path_to_mobile = f'{mobile_folder}{header}_{basename_index_pdb}'
    df_target = get_hb_clashes_df(path_to_target)
    df_mobile = get_hb_clashes_df(path_to_mobile)

    df_both = pd.concat([df_target, df_mobile])


    df_both.to_pickle(output_path_pkl)
    
    # Save the DataFrame to a TSV file
    output_path_tsv = f'{output_folder}{header}_{basename_index}.tsv'
    df_both.to_csv(output_path_tsv, sep='\t', index=False)
    
    print(f'========  {output_path_tsv}  saved =============')
    

def main():
    basenames = [file.name for file in Path(target_folder).glob('*.pdb')] # e.g. ground_truth_00039.pdb

    # Determine the number of available CPU cores
    num_cores = 128

    # Use multiprocessing to parallelize the processing with all available CPU cores
    with Pool(processes=num_cores) as pool:
        pool.map(process_proteins, basenames)

if __name__ == "__main__":
    
    main()

