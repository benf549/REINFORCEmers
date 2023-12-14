import torch
import torch.nn as nn
from utils.build_rotamers import RotamerBuilder
from utils.dataset import BatchData


class ReinforcemerRepacker(nn.Module):
    def __init__(self, node_embedding_dim: int, rbf_encoding_params: dict, **kwargs):
        super(ReinforcemerRepacker, self).__init__()
        self.node_embedding_dim = node_embedding_dim

        # Initialize sequence index encoder.
        self.sequence_index_embedding_layer = nn.Embedding(21, self.node_embedding_dim)
        # Initialize edge distance encoder.
        self.edge_distance_encoding_module = RBF_Encoding(**rbf_encoding_params)
        # Initialize rotamer builder inside model to move helper tensors to correct device as necessary.
        rotamer_builder = RotamerBuilder()
        self.encoder = EncoderModule()
    
    def forward(self, batch: BatchData):

        # Use sequence embeddings as initial embeddings for node message passing.
        sequence_encodings = self.sequence_index_embedding_layer(batch.sequence_indices.long())

        # Encode the edge distances.
        edge_attrs = self.edge_distance_encoding_module(batch.edge_distance)

        raise NotImplementedError

class EncoderModule(nn.Module):
    def __init__(self):
        super(EncoderModule, self).__init__()


class RBF_Encoding(nn.Module):
    """
    Implements the RBF Encoding from ProteinMPNN as a module that can get stored in the model.
    """
    def __init__(self, num_bins, bin_min, bin_max):
        super(RBF_Encoding, self).__init__()
        self.num_bins = num_bins
        self.bin_min = bin_min
        self.bin_max = bin_max
        self.D_sigma =  (bin_max - bin_min) / num_bins
        self.register_buffer('D_mu', torch.linspace(bin_min, bin_max, num_bins).view([1,-1]))

    def forward(self, distances):
        D_expand = torch.unsqueeze(distances, -1)
        RBF = torch.exp(-((D_expand - self.D_mu) / self.D_sigma)**2) + 1E-8
        return RBF