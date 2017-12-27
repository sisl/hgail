
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

'''
Description:
    state: (x postion, desired x position, timestep)
    actions: dx in [-1,1]
    rewards: -l2 from desired x position
    transitions: x = x + dx 
    initial state distribution: x in [-1,1], desired x in [-1,1]
'''
class TwoRoundContinuousDeterministicEnv(Env):

    def __init__(self):
        self._action_space = spaces.Box(low=np.array([-1]), high=np.array([1]))
        self._observation_space = spaces.Box(low=np.array([-1,-1,0]), high=np.array([1,1,2]))
        self.reset()

    def _get_obs(self):
        return self.state

    def reset(self):
        x = (np.random.rand() - .5) * 2
        x_des = (np.random.rand() - .5) * 2
        t = 0
        self.state = [x, x_des, t]
        return self._get_obs()

    def step(self, action):
        assert self.action_space.contains(action)

        x, x_des, t = self.state
        t += 1
        x = np.clip(x + action, -1, 1)
        r = - (x - x_des) ** 2
        done = t > 1
        self.state = [x, x_des, t]
        return self._get_obs(), r, done, {}

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space
