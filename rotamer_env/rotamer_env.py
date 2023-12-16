import numpy as np
import gym
import torch
from gym import spaces
from utils.compute_reward import compute_reward
from IPython import embed


'''
Not used in final project, but completed for future directions
'''

class rotamer_env(gym.Env):
    #makes sure environment doesn't render
    metadata = {"render_modes": None, "render_fps": None}
    num_bins = 72
    size = 0

    def __init__(self, size=10, render_mode=None):
        
        #Define state space. There are 5 states, 1-4 corresponding to the current rotamer to decode, and 5 as a terminal state
        self.observation_space = spaces.Dict({"agent" : spaces.Discrete(5)})

        #Define action space. There are 72 bins, corresponding to 72 rotameric bins
        self.action_space = spaces.Tuple(tuple([spaces.Discrete(72) for _ in range(size)]))
        #self.action_space = spaces.Discrete(72)
        self.size = size

    def _get_obs(self):
        return {"agent": self._agent_location}
    
    #TODO: could return other info, just redundant for now
    def _get_info(self):
        return {"agent": self._agent_location}

    def reset(self, options = None):
        super().reset()

        #Define the initial state of the environment. The agent is initialize as NaNs corresonding to no rotamers decoded
        self._agent_location = np.zeros(5, dtype=int)

        observation = self._get_obs()
        return observation, self._get_info()
    
    def step(self, action, coords = None, bb_bb_eidx = None, bb_label_indices= None):
        #finds first location of NaN which corresponds to the next rotamer to decode
        curr_chi = np.where(self._agent_location == 0)[0][0]
        self._agent_location[curr_chi] = 1

        terminated = not np.any(self._agent_location == 0)
        #does not calculate reward if the state is not terminal since partial building of side chains is both nontrivial and might not make physical sense
        if not terminated:
            reward = 0
        else:
            reward = compute_reward(coords, bb_bb_eidx, bb_label_indices)
        observation = self._get_obs()

        return observation, reward, terminated, self._get_info()