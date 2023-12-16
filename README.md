# reinforcemers

Training logged to:
https://wandb.ai/benf549/reinforcemers?workspace=user-benf549

```
.
├── GAE.py : unused A2C model for PPO loss
├── README.md
├── evaluate_model.py : evaluation scripts
├── files : helper tensors used to build 3d coordinates from model outputs
├── generate_sequence_clusters.py : cluster seqs for train splits
├── rotamer_env : unused Gym Env
├── train_reinforced.py : train pretrained model using REINFORCE
├── train_supervised.py : pretrain model with supervised learning
└── utils
    ├── build_rotamers.py : calculates 3d coords given chi angles
    ├── compute_reward.py : unused reward function for planned PPO model
    ├── constants.py : biophysical constants
    ├── dataset.py : Dataset and Sampler classes for model
    ├── get_rotamers_freq_calcRMSD.py : calc rmsd from outputs to test data
    ├── loss.py : unused PPO loss 
    └── model.py : REINFORCEmer Repacker model
```
