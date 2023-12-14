import torch
import torch.nn as nn
from torch_scatter import scatter
from utils.build_rotamers import RotamerBuilder
from utils.dataset import BatchData


class ReinforcemerRepacker(nn.Module):
    def __init__(
        self, node_embedding_dim: int, edge_embedding_dim: int, num_encoder_layers: int, dropout: float, rbf_encoding_params: dict, 
        **kwargs
    ) -> None:
        super(ReinforcemerRepacker, self).__init__()

        self.node_embedding_dim = node_embedding_dim
        self.edge_embedding_dim = edge_embedding_dim
        self.dropout = dropout

        # sequence index encoder.
        self.sequence_index_embedding_layer = nn.Embedding(21, self.node_embedding_dim)

        # edge distance encoder.
        self.edge_distance_encoding_module = RBF_Encoding(**rbf_encoding_params)
        self.edge_embedding_linear = nn.LazyLinear(self.edge_embedding_dim)

        # encoder layers which will build up the node embeddings.
        self.encoder_layers = nn.ModuleList([EncoderModule(node_embedding_dim, edge_embedding_dim, dropout, **kwargs) for _ in range(num_encoder_layers)])

        # rotamer builder inside model to move helper tensors to correct device as necessary.
        rotamer_builder = RotamerBuilder()
    
    def forward(self, batch: BatchData):
        # Use sequence embeddings as initial embeddings for node message passing.
        bb_nodes = self.sequence_index_embedding_layer(batch.sequence_indices.long())

        # Encode the edge distances.
        edge_attrs = self.edge_distance_encoding_module(batch.edge_distance).flatten(start_dim=1)
        edge_attrs = self.edge_embedding_linear(edge_attrs)

        # Build up node embeddings with successive rounds of message passing.
        for encoder_layer in self.encoder_layers:
            bb_nodes, edge_attrs = encoder_layer(bb_nodes, batch.edge_index, edge_attrs)

        # Generate a mask that can be used with the native 
        raise NotImplementedError
    

class EncoderModule(nn.Module):
    def __init__(self, node_embedding_dim, edge_embedding_dim, dropout, num_attention_heads, use_mean_attention_aggr, **kwargs):
        super(EncoderModule, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(0.2)

        self.nheads = num_attention_heads
        self.use_mean_attention_aggr = use_mean_attention_aggr
        self.mha_aW = torch.nn.ModuleList([torch.nn.Linear((2 * node_embedding_dim) + edge_embedding_dim, node_embedding_dim) for _ in range(self.nheads)])
        self.mha_aA = torch.nn.ModuleList([torch.nn.Linear(node_embedding_dim, 1) for _ in range(self.nheads)])

        self.compute_neighbor_node_update = nn.Sequential(
            nn.Linear(edge_embedding_dim + (2 * node_embedding_dim), node_embedding_dim),
            nn.GELU(), 
            nn.Linear(node_embedding_dim, node_embedding_dim), 
            nn.GELU(), 
            nn.Linear(node_embedding_dim, node_embedding_dim)
        )

        self.compute_dense_node_update = nn.Sequential(
            nn.Linear(node_embedding_dim, 4 * node_embedding_dim),
            nn.GELU(),
            nn.Linear(4 * node_embedding_dim, node_embedding_dim),
        )

        self.compute_edge_update = nn.Sequential(
            nn.Linear(edge_embedding_dim + (2 * node_embedding_dim), edge_embedding_dim),
            nn.GELU(),
            nn.Linear(edge_embedding_dim, edge_embedding_dim),
            nn.GELU(),
            nn.Linear(edge_embedding_dim, edge_embedding_dim)
        )

        self.attention_aggr_linear = None
        if not self.use_mean_attention_aggr:
            self.attention_aggr_linear = nn.Linear(self.nheads * node_embedding_dim, node_embedding_dim)

        self.node_norm1 = nn.LayerNorm(node_embedding_dim)
        self.node_norm2 = nn.LayerNorm(node_embedding_dim)
        self.edge_norm = nn.LayerNorm(edge_embedding_dim)


    def forward(self, bb_nodes, eidx, eattr):

        # Expand the node embeddings to match the number of edges.
        source_nodes = bb_nodes[eidx[0]]
        sink_nodes = bb_nodes[eidx[1]]

        # Concatenate the source and sink nodes with the edge attributes.
        bb_node_update = torch.cat([source_nodes, sink_nodes, eattr], dim=1)

        # Compute GATv2 multi-head attention weights.
        bb_edges_mh_atten = torch.stack([aA(self.leakyrelu(aW(bb_node_update))) for aA, aW in zip(self.mha_aA, self.mha_aW)])

        # Apply 3-Layer MLP to compute node updates per edge.
        bb_node_update = self.compute_neighbor_node_update(bb_node_update)

        # Numerically stable (manual) softmax computation for each attention head for each neighborhood.
        atten_scatter_max = scatter(bb_edges_mh_atten, eidx[1], dim=1, reduce='max', dim_size=bb_nodes.shape[0])
        atten = torch.exp(bb_edges_mh_atten - atten_scatter_max[:, eidx[1]])
        atten_norm = scatter(atten + 1e-12, eidx[1], dim=1, reduce='sum', dim_size=bb_nodes.shape[0])
        atten = atten / atten_norm[:, eidx[1]]
        atten = self.dropout(atten)

        # Apply softmaxed attention coefficients to weight node updates.
        bb_node_update = atten[:, eidx[1]] * bb_node_update.unsqueeze(0).expand(atten.shape[0], -1, -1)

        # Sum attention-weighted node updates for final node updates.
        bb_nodes_updated = scatter(bb_node_update, eidx[1], dim=1, reduce='sum', dim_size=bb_nodes.shape[0])

        # Aggregate multi-head attention updates.
        if self.attention_aggr_linear is None:
            bb_nodes_updated = bb_nodes_updated.mean(dim=0)
        else:
            bb_nodes_updated = bb_nodes_updated.transpose(0, 1).reshape(bb_nodes_updated.shape[1], -1)
            bb_nodes_updated = self.attention_aggr_linear(bb_nodes_updated)

        # Update node embeddings and normalize.
        bb_nodes = self.node_norm1(bb_nodes + self.dropout(bb_nodes_updated))
        bb_nodes = self.compute_dense_node_update(bb_nodes)
        bb_nodes = self.node_norm1(bb_nodes + self.dropout(bb_nodes_updated))

        # Update edge embeddings and normalize using updated bb_nodes.
        source_nodes = bb_nodes[eidx[0]]
        sink_nodes = bb_nodes[eidx[1]]
        edge_update = torch.cat([source_nodes, sink_nodes, eattr], dim=1)
        edge_update = self.compute_edge_update(edge_update)
        eattr = self.edge_norm(eattr + self.dropout(edge_update))

        return bb_nodes, eattr


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