import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions

import matplotlib.pyplot as plt
import numpy as np
import gym

from utils.model import ReinforcemerRepacker

'''
Not used in final project, but completed for future directions
'''

LEARNING_RATE = 0.0005

env = gym.make("rotamer_env/rotamer_env-v0")

class ActorCritic(nn.Module):
    def __init__(self, actor, critic):
        super().__init__()

        self.actor = actor
        self.critic = critic

    def forward(self, state):
        action_pred = self.actor(state)
        value_pred = self.critic(state)

        return action_pred, value_pred

#add pretrained model
actor = ReinforcemerRepacker()
critic = ReinforcemerRepacker()

policy = ActorCritic(actor, critic)


optimizer = optim.Adam(policy.parameters(), lr = LEARNING_RATE)