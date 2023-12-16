import os
import torch
import numpy as np
from collections import defaultdict
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
from tqdm import tqdm

from .constants import aa_short_to_idx


@np.vectorize
def map_index(char: str) -> int:
    """
    Takes an MSA character and maps it to an index in a probability vector.
    """
    if char == '-':
        return -1
    else:
        return aa_short_to_idx[char] if char in aa_short_to_idx else aa_short_to_idx['X']


@dataclass
class MSA_Data:
    # Numpy array as array of strings of same length.
    msa: np.ndarray

    def get_query_sequence(self):
        """
        Returns the query sequence that generated the MSA.
        """
        return self.msa[0]

    def msa_to_torch_tensor(self) -> torch.Tensor:
        """
        Convert the numpy array of strings to a torch tensor of shape (N x 21) probability vectors.
        """
        # Convert the array of strings to a matrix of itegers.
        msa_mtx = np.array([list(seq) for seq in self.msa])
        index_array = map_index(msa_mtx)

        # Convert the matrix of integers to a matrix of probability vectors.
        output = []
        for col in index_array.T:
            frequencies = np.zeros(21)
            indices, counts = np.unique(col, return_counts=True)

            # Handle -1 (gaps) by removing them from the counts and indices.
            if indices[0] == -1:
                indices = indices[1:]
                counts = counts[1:]

            # Convert counts to frequencies.
            frequencies[indices] = counts
            frequencies /= np.sum(frequencies)

            output.append(torch.tensor(frequencies))
        output = torch.stack(output)

        return output

    def __repr__(self) -> str:
        return f"<MSA Query {self.shape()}: {self.get_query_sequence()[:30]}...>"
    
    def __str__(self) -> str:
        return f"<MSA Query {self.shape()}: {self.get_query_sequence()[:30]}...>"

    def shape(self) -> Tuple[int, int]:
        """
        Returns (depth, query length) of the MSA.
        """
        return len(self.msa), len(self.msa[0])

def load_msa_from_disk(msa_data_paths: dict) -> Optional[Dict[str, MSA_Data]]:
    output = {}
    try:
        for chain_id, msa_path in msa_data_paths.items():
            msa = np.load(msa_path)
            output[chain_id] = MSA_Data(msa)
    except FileNotFoundError:
        return None

    return output


class MultipleSequenceAlignmentDataset():
    """
    A class representing the OpenFold BFD dataset of multiple sequence alignments (MSAs) for protein structures.

    Args:
        params (dict): A dictionary containing the parameters for initializing the dataset.

    Attributes:
        all_data (dict): A dictionary that maps from PDB ID to a dictionary mapping chain IDs to paths.

    Methods:
        get_pdb_msas: Retrieve the corresponding dict for a given PDB ID code.
    """

    def __init__(self, params: dict):
        """
        Initialize the MultipleSequenceAlignmentDataset.

        Args:
            params (dict): A dictionary containing the parameters for initializing the dataset.
        """

        pdb_map = {
            tuple(x.rsplit('/', 1)[-1].replace('_seqs.npy', '').split('_')): os.path.join(params['path_to_msa_data'], x)
            for x in os.listdir(params['path_to_msa_data']) if x.endswith('_seqs.npy')
        }

        # Create the same dictionary that maps from pdb_id to a dict of chain_id paths:
        self.all_data = defaultdict(dict)
        for (pdb_id, chain_id), path in pdb_map.items():
            self.all_data[pdb_id][chain_id] = path

        # Point known missing chains to the dictionary:
        missing = 0
        with open(params['path_to_duplicate_chain_file'], 'r') as f:
            # Each line is a space-separated list of pdb_chain identifiers
            for line in f.readlines():
                representative = None
                pdb_chains = []

                # Split by pdb_chain identifier and loop until we find the pdb_chain we have data for.
                for pdb in line.split():
                    pdbid, chain = pdb.split('_')
                    if pdbid in self.all_data and chain in self.all_data[pdbid]:
                        representative = (pdbid, chain)
                    pdb_chains.append((pdbid, chain))

                # If we have a representative, point all other chains to the representative.
                if representative is not None:
                    representative_pdb, representative_chain = representative
                    for pdb_other, chain_other in pdb_chains:
                        self.all_data[pdb_other][chain_other] = self.all_data[representative_pdb][representative_chain]
                else:
                    # Possible if there were hits in another sequence database but not the BFD.
                    missing += 1

        self.all_data = dict(self.all_data)
        print(f"Loaded sequence metadata for {len(self.all_data)} unique chains.")
        print(f"Missing MSAs for {missing} unique chains (construct numpy arrays for other datasets to default to).")
    
    def get_pdb_msas(self, pdb_id: str) -> Optional[Dict[str, str]]:
        """
        Retrieve the multiple sequence alignments (MSAs) for a given PDB ID code.

        Args:
            pdb_id (str): The PDB ID for which to retrieve the MSAs.

        Returns:
            dict: A dict mapping original chain ID to path to MSAs associated with that chain.
        """
        if pdb_id in self.all_data:
            return self.all_data[pdb_id]
        else:
            return None