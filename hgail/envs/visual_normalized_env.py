
from collections import deque
import numpy as np
import skimage.transform

from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.envs.proxy_env import ProxyEnv

from sandbox.rocky.tf import spaces

class VisualNormalizedEnv(ProxyEnv, Serializable):

    def __init__(
            self, 
            env,
            size=(84,84),
            history_len=4,
            mean=.15):
        Serializable.quick_init(self, locals())
        ProxyEnv.__init__(self, env)
        self.size = size
        self.history_len = history_len
        self.mean = mean
        self.obs_shape = self.size + (self.history_len,)
        self.history = np.zeros(self.obs_shape)
        self._observation_space = spaces.Box(-np.inf, np.inf, shape=self.obs_shape)

    def _normalize_obs(self, obs):
        obs = skimage.transform.resize(obs, self.size, mode='constant')
        # obs originally in range [0-255], so dividing brings to [0,1]
        # self.mean is wrt that range
        obs = obs / 255. - self.mean
        return obs

    def _process_obs(self, obs):
        obs = self._normalize_obs(obs)
        self.history[...,1:] = self.history[...,:-1]
        self.history[...,0] = obs
        return np.copy(self.history)

    def reset(self):
        self.history.fill(0)
        obs = self._wrapped_env.reset()
        return self._process_obs(obs)

    def step(self, action):
        next_obs, reward, done, info = self._wrapped_env.step(action)
        next_obs = self._process_obs(next_obs)
        return Step(next_obs, reward, done, **info)

    @property 
    def observation_space(self):
        return self._observation_space

visual_normalized_env = VisualNormalizedEnv
