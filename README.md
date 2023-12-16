
# REINFORCEmers: Protein sidechain packing using a biophysically motivated reward function

WandB Session with Logged Training Runs: https://wandb.ai/benf549/reinforcemers?workspace=user-benf549

## Reinforcemer GNN Model Description
We implemented a graph-transformer inspired by the ProteinMPNN
architecture to predict amino acid sidechain orientation given fixed backbone coordinates and
amino acid identities. The architecture has 3 encoder layers which build up a latent representation
for each protein node using GATv2 attention-weighted local neighborhood message passing.
The encoder-generated node representations are used by 4 chi-angle decoding layers which predict
the sidechain orientation angles of a given node while considering the ground-truth orientations of
adjacent residues in the one-hop neighborhood of the K-Nearest Neighbors (KNN) graph. Care
is taken to ensure ground-truth residue orientations do not get passed into the nodes during the
message-passing process which greatly speeds up training as a form of teacher-forcing as is often
used when training transformer models. The input to the model is the the protein backbone coordinates
and the output is a 3D structure with the sidechains grafted onto the backbone. KNN Graphs are
constructed from every protein complex’s 3D atomic coodinates with nodes coarse-grained into
backbone coordinate frames (the N/Cα/C/O atoms shared by all amino acids which get linked
together into a protein). Node edges are featurized by the distances between connect the nodes with
edges featurized by distances encoded with Gaussian smearing/radial basis functions.
We will consider a particular choice of model weights a policy that operates on states defined for
each amino acid node and its local context within the KNN graph. This inherently high-dimensional
state-space necessitates the GNN function class for policy development. The actions under this
formulation are the choices for χ angles given the current state.
This stochastic policy generates probabilities of actions π(a|s) in the multinomial distribution of χ
angles of discretized into 5◦ bins. Rather than performing regression directly for the chi angle(s) of
interest we formulate the problem with discrete bins since the latent distribution of chi angles may
be multimodal and discretization enables sampling from areas with higher probability density in the
learned distribution. To take advantage of the radial symmetry of angles, we can encode our angles in
Gaussian smearing/radial basis functions similar to our edge distances, but cyclically wrap last bin
around to the first, bin ensuring that densities are preserved. This cyclically-symmetric encoding can
then be used to concatenate the encoded representation of a current node to any previously-predicted
chi angles in order to autoregressively predict the next node. We supply the logits produced by the model as inputs to a cross-entropy function for supervised pretraining learning with the target χ
angles encoded in the same manner.
By predicting sidechain coordinates for all amino acids in a protein, we can construct 3D point
clouds implied by the chosen chi angles evaluate these generated point clouds to compute a reward.

## Reinforcemers file structure
```
.
├── GAE.py : unused A2C model for PPO loss
├── README.md
├── evaluate_model.py : evaluation scripts
├── files : helper functions and data for model
├── generate_sequence_clusters.py : cluster seqs for train splits
├── rotamer_env : unused Gym Env
├── train_reinforced.py : train pretrained model using REINFORCE
├── train_supervised.py : pretrain model
└── utils
    ├── build_rotamers.py : calculates atom coords given chi angles
    ├── compute_reward.py : reward function for REINFORCE model
    ├── constants.py : biophysical constants
    ├── dataset.py : Dataset and Sampler classes for model
    ├── get_rotamers_freq_calcRMSD.py : calc rmsd from outputs to test data
    ├── loss.py : unused PPO loss 
    └── model.py : REINFORCEmer Repacker model
```
