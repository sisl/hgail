
import numpy as np

from rllab.envs.normalized_env import normalize as normalize_env

class VectorizedNormalizedEnv(normalize_env):

    def _update_obs_estimate(self, obs):
        # assert (n_envs, obs_dim) shape, i.e., already flat
        assert len(obs.shape) == 2
        self._obs_mean = (1 - self._obs_alpha) * self._obs_mean + self._obs_alpha * np.mean(obs, axis=0)
        self._obs_var = (1 - self._obs_alpha) * self._obs_var + self._obs_alpha * np.mean(np.square(obs - self._obs_mean), axis=0)

    @property
    def vectorized(self):
        return True

    def vec_env_executor(self, n_envs, max_path_length):
        self.num_envs = n_envs
        return self

vectorized_normalized_env = VectorizedNormalizedEnv