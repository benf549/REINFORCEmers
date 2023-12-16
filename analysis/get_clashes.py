import pandas as pd
import os
import subprocess
from collections import defaultdict





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




PROBE_CMD = '/programs/x86_64-linux/probe/2.16.130520/probe -U -CON -Explicit -NOFACE -WEAKH -DE32 -WAT2wat -4 -ON -MC ALL ALL -'
pdb_file = './jerry_test_proteins_noxaa/repacked/00000.pdb'
with open(pdb_file) as f:
    pdb = f.read()
        
probe_out = subprocess.run(PROBE_CMD.split(), input=pdb, text=True, capture_output=True).stdout

contacts_df = parse_probe_output(probe_out)


## compute clashes
hbonding_dict = defaultdict(lambda: defaultdict(set))
clashing_dict = defaultdict(set)
for row in contacts_df.itertuples():
    if row.contact_type == 'hb':
        atom2 = (row.chain2, row.resname2, row.resnum2, row.name2)
        hbonding_dict[row.chain1, row.resnum1][row.name1].add(atom2)
    elif row.contact_type == 'bo':
        clashing_dict[row.chain1, row.resnum1].add(row.name1)