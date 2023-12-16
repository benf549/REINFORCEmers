import torch
import torch.nn.functional as F
import torchrl
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
)
import gym
from rotamer_env.rotamer_env import rotamer_env
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.envs.libs.gym import GymEnv, GymWrapper

from IPython import embed

'''
Start of PPO loss function, not used in final project
'''
SUB_BATCH_SIZE = 64
NUM_EPOCHS = 10
#clip value for PPO loss
CLIP_EPSILON = (0.2)
#reward discount factor for GAE
GAMMA = 0.99
#bias-variance tradeoff for GAE
LAMBDA = 0.95
ENTROPY_EPS = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def transform_env(base_env:gym.Env):
    trans_env = TransformedEnv(
        base_env,
        Compose(
            StepCounter(max_steps=5)
        )
    )
def compute_loss():
    raise NotImplementedError
