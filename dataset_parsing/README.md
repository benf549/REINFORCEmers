# Reparsing of the PDB with rotamer flipping and protonation by `reduce`.
Benjamin Fry (bfry@g.harvard.edu)

Note: Make sure ProDy>=2.4 is installed since the mmCIF parsing seems to be broken in earlier verisons.

## Dataset setup:
### Download the mmCIF-formatted PDB from RCSB website using RSYNC and wget additional metadata files:

```bash
rsync -rlpt -v -z --delete --port=33444 rsync.wwpdb.org::ftp/data/data/assemblies/mmCIF/divided/ /scratch/bfry/mmCIF_bioasmb/
wget https://files.rcsb.org/pub/pdb/derived_data/index/entries.idx
```

Also install `reduce` from the [GitHub repository](https://github.com/rlabduke/reduce) and run its `python2 update_het_dict.py` script after installation to update the heteroatom dictionary.

### Filters the dataset by size and converts every biological assembly to a pdb file while storing modifications.
Outputs generated PDB files to `/scratch/bfry/filtered_pdb_bioasmb/`
* Use 10_000 residues as the first cutoff.
* Only run on XRAY and CRYO structures < 3.5 Angstrom Resolution.
* Only keep structures deposited before 06/01/2022.

Generates a `./filtering_metadata.pkl` file in working directory containing information about which PDB files were actually parsed and pickles of all their header information as well as recording the segment-id/chain-id/resnum remapping which is necesary for running.

```bash
python prepare_reduce_inputs.py
```

## Running `reduce`:
### Run `reduce` on the filtered PDB files to add hydrogens and flip rotamers.
Outputs generated PDB files to `/scratch/bfry/reduce_filtered_pdb_bioasmb/`

Generates a `./reduce_metadata.pkl` file tracking whether reduce ran successfully merged with the previous metadata pickle file.
Outputs are generated with completely different segment/chain/resnum mapping compared to what might be expected at the beginning of the pipeline.

```bash
python run_reduce_parallel.py
```

## Dataset processing & filtering:

###### TODO:
- [ ] Record all heteroatom/nucleic acid/peptide/pTM locations and create masks to avoid them.
- [ ] Sanity check inputs for clashes.
- [ ] Make any chain with < 40 residues a peptide.
- [ ] Parse out hydrogen locations for Cys, Thr, Ser, Tyr, His sidechains in addition to all residue matrices.

```bash
python reparse_pdb.py
```
