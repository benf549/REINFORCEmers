# reinforcemers
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
