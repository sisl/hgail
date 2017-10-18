
import gym
from gym import spaces
from gym.utils import seeding

from rllab.envs.base import Env
from rllab import spaces

import numpy as np

class TwoRoundNondeterministicRewardEnv(Env):
    def __init__(self):
        self.reset()

    def step(self, action):
        rewards = [
            [
                [-1, 1], # expected value 0
                [0, 0, 9] # expected value 3. This is the best path.
            ],
            [
                [0, 2], # expected value 1
                [2, 3] # expected value 2.5
            ]
        ]

        assert self.action_space.contains(action)

        if self.firstAction is None:
            self.firstAction = action
            reward = 0
            done = False
        else:
            reward = np.random.choice(rewards[self.firstAction][action])
            done = True

        return self._get_obs(), reward, done, {}

    def reset(self):
        self.firstAction = None
        return self._get_obs()

    def _get_obs(self):
        if self.firstAction is None:
            return 2
        else:
            return self.firstAction
    
    @property
    def action_space(self):
        return spaces.Discrete(2)

    @property
    def observation_space(self):
        return spaces.Discrete(3)
