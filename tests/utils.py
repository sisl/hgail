
import numpy as np

from rllab.envs.base import Env

from sandbox.rocky.tf import spaces

class MockPolicy(object):

    def __init__(self, action_space):
        self.action_space = action_space
        self.recurrent = False
        self.counter = None

    def reset(self, dones=None):
        if self.counter is None:
            self.counter = np.zeros(len(dones))
        for i, done in enumerate(dones):
            if done:
                self.counter[i] = 0

    def get_actions(self, observations):
        n_samples = len(observations)
        actions = [self.action_space.sample() for i in range(n_samples)]
        agent_infos = dict(
            prob=[[1,0] for _ in range(n_samples)],
            latent=[[self.counter[i]] for i in range(n_samples)],
            latent_info=dict(
                prob=[[0,1] for _ in range(n_samples)],
                latent=[[self.counter[i]] for i in range(n_samples)],
                update=[[self.counter[i] % 2 == 0] for i in range(n_samples)],
                latent_info=dict(
                    latent=[[self.counter[i] * 2]  for i in range(n_samples)]
                )
            )
        )
        self.counter += 1
        return actions, agent_infos

    @property
    def distribution(self):
        return MockDist()

class MockAlgo(object):
    def __init__(
            self, 
            policy=None, 
            env=None, 
            baseline=None, 
            discount=1.,
            batch_size=10,
            max_path_length=1000,
            gae_lambda=1.
        ):
        self.policy = policy
        self.env = env
        self.baseline = baseline
        self.discount = discount
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.gae_lambda = gae_lambda
        self.center_adv = False
        self.positive_adv = False

class MockDist(object):

    def entropy(self, *args):
        return 0

class MockEnv(Env):
    def __init__(self, observation_space, action_space):
        self._observation_space = observation_space
        self._action_space = action_space
    def reset(self):
        return self.observation_space.sample()
    def step(self, action):
        assert self.action_space.contains(action)
        next_obs = self.observation_space.sample()
        return next_obs, 0, False, {}
    @property
    def observation_space(self):
        return self._observation_space
    @property
    def action_space(self):
        return self._action_space

class MockEnvSpec(object):
    def __init__(
            self,
            observation_space=spaces.Box(np.array([0]),np.array([1])),
            action_space=spaces.Discrete(2),
            num_envs=1):
        self.observation_space = observation_space
        self.action_space = action_space
        self.num_envs = num_envs
