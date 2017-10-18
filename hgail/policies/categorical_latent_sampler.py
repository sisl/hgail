
import copy
import numpy as np
import tensorflow as tf

from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides

from hgail.policies.categorical_latent_var_mlp_policy import CategoricalLatentVarMLPPolicy
from hgail.policies.latent_sampler import LatentSampler

class CategoricalLatentSampler(LatentSampler, CategoricalLatentVarMLPPolicy):

    def __init__(
            self, 
            scheduler,
            max_n_envs=20,
            **kwargs):
        Serializable.quick_init(self, locals())
        super(CategoricalLatentSampler, self).__init__(**kwargs)
        self._scheduler = scheduler
        self._latent_values = np.zeros((max_n_envs, self.action_space.n))

    @overrides
    def get_action(self, flat_obs):

        # sample and concat latent var
        latent, latent_info = self.latent_sampler.get_action(flat_obs)
        flat_obs = np.hstack((flat_obs, latent))
        
        # if scheduler says to update, then fprop the observation
        # otherwise, just use the previous values
        should_update = self._scheduler.should_update(flat_obs)
        prob = np.zeros(self.dim)
        if should_update:
            prob = self._f_prob([flat_obs])[0]
            self._latent_values[0] = self.action_space.flatten(
                self.action_space.weighted_sample(prob))

        # package agent infos with latent info
        agent_infos = dict(
            prob=prob, 
            update=[should_update], 
            latent=copy.deepcopy(self._latent_values[0]),
            latent_info=latent_info
        )
            
        return copy.deepcopy(self._latent_values[0]), agent_infos

    @overrides
    def get_actions(self, flat_obs):

        # sample and concat latent var
        latent, latent_info = self.latent_sampler.get_actions(flat_obs)
        flat_obs = np.hstack((flat_obs, latent))

        # selectively update the actions for the different envs
        update_indicators = self._scheduler.should_update(flat_obs)
        probs = self._f_prob(flat_obs)
        actions = list(map(self.action_space.weighted_sample, probs))

        # flatten because no interaction with envs
        actions = self.action_space.flatten_n(actions)
        update = [False] * len(self._latent_values)
        for (i, indicator) in enumerate(update_indicators):
            if indicator:
                self._latent_values[i] = actions[i]

        # package agent infos with latent info
        agent_infos = dict(
            prob=probs, 
            update=update_indicators.astype(bool), 
            latent=copy.deepcopy(self._latent_values),
            latent_info=latent_info
        )    

        return copy.deepcopy(self._latent_values), agent_infos

    @overrides
    def reset(self, dones=None):
        dones = [True] if dones is None else dones
        self._latent_values = self._latent_values[:len(dones)]
        self._scheduler.reset(dones)
        self.latent_sampler.reset(dones)

    @overrides
    @property
    def state_info_specs(self):
        return [('latent', (self.latent_sampler.dim,))]

    def __getstate__(self):
        d = CategoricalLatentVarMLPPolicy.__getstate__(self)
        e = LatentSampler.__getstate__(self)
        d.update(e)
        d['_latent_values'] = self._latent_values
        d['_scheduler'] = self._scheduler
        return d

    def __setstate__(self, d):
        CategoricalLatentVarMLPPolicy.__setstate__(self, d)
        LatentSampler.__setstate__(self, d)
        self._scheduler = d['_scheduler']
        self._latent_values = d['_latent_values']

