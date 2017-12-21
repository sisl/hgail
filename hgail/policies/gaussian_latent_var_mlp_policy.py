
import numpy as np
import tensorflow as tf

from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides

import sandbox.rocky.tf.core.layers as L
from sandbox.rocky.tf.core.network import MLP
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.spaces.box import Box

class GaussianLatentVarMLPPolicy(GaussianMLPPolicy):
    def __init__(
            self,
            name,
            env_spec,
            latent_sampler,
            hidden_sizes=(32, 32),
            std_hidden_sizes=(32, 32),
            min_std=1e-6,
            std_hidden_nonlinearity=tf.nn.tanh,
            hidden_nonlinearity=tf.nn.tanh,
            output_nonlinearity=None
    ):
        Serializable.quick_init(self, locals())
        assert isinstance(env_spec.action_space, Box)
        self.latent_sampler = latent_sampler

        with tf.variable_scope(name):
            obs_dim = env_spec.observation_space.flat_dim + self.latent_sampler.dim
            action_dim = env_spec.action_space.flat_dim

            # create networks
            mean_network = MLP(
                name="mean_network",
                input_shape=(obs_dim,),
                output_dim=action_dim,
                hidden_sizes=hidden_sizes,
                hidden_nonlinearity=hidden_nonlinearity,
                output_nonlinearity=output_nonlinearity,
            )
            std_network = MLP(
                name="std_network",
                input_shape=(obs_dim,),
                input_layer=mean_network.input_layer,
                output_dim=action_dim,
                hidden_sizes=std_hidden_sizes,
                hidden_nonlinearity=std_hidden_nonlinearity,
                output_nonlinearity=None,
            )
            
            super(GaussianLatentVarMLPPolicy, self).__init__(
                name=name, 
                env_spec=env_spec, 
                mean_network=mean_network,
                std_network=std_network
            )

    @overrides
    def get_action(self, observation):
        flat_obs = self.observation_space.flatten(observation)

        # sample and concat latent var
        latent, latent_info = self.latent_sampler.get_action(flat_obs)
        flat_obs = np.hstack((flat_obs, latent))

        # compute mean and log_std and sample action
        mean, log_std = [x[0] for x in self._f_dist([flat_obs])]
        rnd = np.random.normal(size=mean.shape)
        action = rnd * np.exp(log_std) + mean
        return action, dict(mean=mean, log_std=log_std, latent=latent, latent_info=latent_info)

    @overrides
    def get_actions(self, observations):
        flat_obs = self.observation_space.flatten_n(observations)

        # sample and concat latent var
        latent, latent_info = self.latent_sampler.get_actions(flat_obs)
        flat_obs = np.hstack((flat_obs, latent))

        # compute means and log_stds, and sample actions
        means, log_stds = self._f_dist(flat_obs)
        rnd = np.random.normal(size=means.shape)
        actions = rnd * np.exp(log_stds) + means
        return actions, dict(mean=means, log_std=log_stds, latent=latent, latent_info=latent_info)

    @overrides
    def reset(self, dones=None):
        self.latent_sampler.reset(dones)

    @property
    def vectorized(self):
        return True

    @overrides
    def dist_info_sym(self, obs_var, state_info_vars=None):
        obs_var = tf.cast(obs_var, tf.float32)
        # if empty dictionary then the latent variable has already been added
        if state_info_vars is not None and len(state_info_vars.keys()) != 0:
            obs_var = self.latent_sampler.merge_sym(obs_var, state_info_vars)
        mean_var, log_std_var = L.get_output([self._l_mean, self._l_std_param], obs_var)
        log_std_var = tf.maximum(log_std_var, self.min_std_param)
        return dict(mean=mean_var, log_std=log_std_var)

    @overrides
    def dist_info(self, obs, state_infos):
        obs = self.latent_sampler.merge(obs, state_infos)
        means, log_stds = self._f_dist(obs)
        return dict(mean=means, log_std=log_stds)

    @overrides
    @property
    def state_info_specs(self):
        return [('latent', (self.latent_sampler.dim,))]
