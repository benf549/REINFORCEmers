import torch
import torch.nn.functional as F



def compute_loss():
    raise NotImplementedError

'''
need to sample from action space, convert to degrees, build rotamers, compute distance matrix, then feed to compute_rotamer_clash_penalty
'''