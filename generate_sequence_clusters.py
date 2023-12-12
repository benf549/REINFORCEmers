#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path
from utils.dataset import UnclusteredProteinDataset, ClusteredDatasetSampler
from utils.constants import MAX_PEPTIDE_LENGTH
from torch.utils.data import DataLoader

def chain_identifier_to_polymer_sequence(protein_data, pdb_code):
    output = {}
    for chain_id, chain_data in protein_data.items():
        chain_sequence = chain_data['polymer_seq']
        if not chain_sequence is None and len(chain_sequence) > MAX_PEPTIDE_LENGTH:
            output['-'.join([pdb_code] + list(chain_id))] = chain_sequence
    return output


def write_polymer_seqs_to_fasta(params):
    all_sequences = {}
    protein_dataset = UnclusteredProteinDataset(params)

    # sampler = ClusteredDatasetSampler(protein_dataset, params)
    # dataloader = DataLoader(protein_dataset, batch_size=10_000, sampler=sampler)
    # for data in dataloader:
    #     print(data)

    for protein_data, pdb_code in protein_dataset:
        polymer_sequence_map = chain_identifier_to_polymer_sequence(protein_data, pdb_code.replace('.pt', ''))
        all_sequences.update(polymer_sequence_map)

    # with open(params['fasta_output_path'], 'w') as f:
    #     for identifier, sequence in all_sequences.items():
    #         f.write(f'>{identifier}\n{sequence}\n')
    
    # # Cluster at 30% sequence identity.
    # # mmseqs easy-cluster fasta.txt cluster30test tmp30test --min-seq-id 0.3 -c 0.5 --cov-mode 5 --cluster-mode 3
    # output_path = params["fasta_output_path"]
    # clustering_output_path_prefixes = os.path.join(params["clustering_output_path"], params['clustering_output_prefix'])
    # mmseq_metadata_output_path = os.path.join(params["clustering_output_path"], "mmseq_tmp_dir")
    # subprocess.run(f'mmseqs easy-cluster {output_path} {clustering_output_path_prefixes} {mmseq_metadata_output_path} --min-seq-id 0.3 -c 0.5 --cov-mode 5 --cluster-mode 3', shell=True)


if __name__ == "__main__":
    params = {
        'dataset_path': '/scratch/bfry/torch_bioasmb_dataset/aa/',
        'clustering_output_prefix': 'torch_bioas_cluster30',
        'clustering_output_path': (output_path := '/scratch/bfry/bioasmb_dataset_sequence_clustering/'),
        'fasta_output_path': os.path.join(output_path, 'all_torch_polymer_sequences.fasta')
    }
    write_polymer_seqs_to_fasta(params)