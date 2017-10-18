
import sandbox.rocky.tf.core.layers as L
from sandbox.rocky.tf.core.network import MLP
from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.misc.overrides import overrides
from rllab.core.serializable import Serializable

import numpy as np
import tensorflow as tf

class CategoricalLatentVarMLPPolicy(CategoricalMLPPolicy):
    def __init__(
            self,
            policy_name,
            env_spec,
            latent_sampler,
            hidden_sizes=(32,32),
            hidden_nonlinearity=tf.nn.tanh,
            prob_network=None):
        Serializable.quick_init(self, locals())
        name = policy_name
        self.latent_sampler = latent_sampler

        with tf.variable_scope(name):
            if prob_network is None:
                input_dim = env_spec.observation_space.flat_dim + self.latent_sampler.dim
                l_input = L.InputLayer(shape=(None, input_dim), name="input")
                prob_network = MLP(
                    input_layer=l_input,
                    output_dim=env_spec.action_space.n,
                    hidden_sizes=hidden_sizes,
                    hidden_nonlinearity=hidden_nonlinearity,
                    output_nonlinearity=tf.nn.softmax,
                    name="prob_network"
                )
                self._output = prob_network.output
                self._inputs = prob_network.input_var

        super(CategoricalLatentVarMLPPolicy, self).__init__(
            name=name, 
            env_spec=env_spec, 
            prob_network=prob_network
        )

    @overrides
    def get_action(self, observation):
        # flatten obs before incorporating latent var
        flat_obs = self.observation_space.flatten(observation)

        # sample and concat latent var
        latent, latent_info = self.latent_sampler.get_action(flat_obs)
        flat_obs = np.hstack((flat_obs, latent))

        # compute probs and action
        prob = self._f_prob([flat_obs])[0]
        action = self.action_space.weighted_sample(prob)

        # return latent_info and latent with probs as agent_infos
        return action, dict(prob=prob, latent=latent, latent_info=latent_info)

    @overrides
    def get_actions(self, observations):
        # flatten obs before incorporating latent var
        flat_obs = self.observation_space.flatten_n(observations)

        # sample and concat latent var
        latent, latent_info = self.latent_sampler.get_actions(flat_obs)
        flat_obs = np.hstack((flat_obs, latent))

        # compute probs and actions
        probs = self._f_prob(flat_obs)
        actions = list(map(self.action_space.weighted_sample, probs))

        # return latent_info and latent with probs as agent_infos
        return actions, dict(prob=probs, latent=latent, latent_info=latent_info)

    @overrides
    def reset(self, dones=None):
        self.latent_sampler.reset(dones)

    @overrides
    def dist_info_sym(self, obs_var, state_info_vars):
        obs_var = tf.cast(obs_var, tf.float32)
        obs_var = self.latent_sampler.merge_sym(obs_var, state_info_vars)
        return dict(prob=L.get_output(self._l_prob, {self._l_obs: obs_var}))

    @overrides
    def dist_info(self, obs, state_infos):
        obs = self.latent_sampler.merge(obs, state_infos)
        return dict(prob=self._f_prob(obs))

    @overrides
    @property
    def state_info_specs(self):
        return [('latent', (self.latent_sampler.dim,))]

    def __getstate__(self):
        return super(CategoricalLatentVarMLPPolicy, self).__getstate__()
