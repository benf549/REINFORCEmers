import torch
from torch import nn
import torchrl
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tensordict.nn import TensorDictModule

import tqdm
import gym
import rotamer_env.rotamer_env as rotamer_env

''''
I know this is bad but so is the documentation for this library

TODO: write value net and GAE
'''

value_net = nn.Sequential(


)