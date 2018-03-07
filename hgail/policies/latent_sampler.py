
from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides

import copy
import numpy as np
import tensorflow as tf

class LatentSampler(object):
    '''
    Mixin class to be used when making a class intended to sample latent variables.

    Since this is a mixin, we add the **kwargs and super call.
    '''
    def __init__(
            self, 
            name, 
            dim, 
            latent_name='latent',
            **kwargs):
        super(LatentSampler, self).__init__(**kwargs)
        self.name = name
        self.dim = dim
        self.latent_name = latent_name
        self._build()

    @property
    def vectorized(self):
        return True

    @property
    def state_info_specs(self):
        '''
        All the inheriting classes can use this because we handle the setting up 
        of the paths separately such that the optimizers think this is the only
        additional state information needed.
        '''
        return [(self.latent_name, (self.dim,))]

    def merge_sym(self, obs_var, state_info_vars=None):
        '''
        Symbolically merges the input variable with the latent variable of the sampler

        Args:
            - obs_var: symbolic variable to merge with, shape = (?, dim)
            - state_info_vars: dictionary containing symbolic variables
                relevant to this latent sampler
        '''
        with tf.variable_scope(self.name):
            if state_info_vars is not None and self.latent_name in state_info_vars.keys():
                latent = state_info_vars[self.latent_name]
            else:
                latent = self.latent
            merged = tf.concat([obs_var, latent], axis=-1)
        return merged

    def merge(self, obs, state_infos):
        '''
        Numeric equivalent to merge_sym - combines obs and state_infos

        Args:
            - obs: observation
            - state_infos: dict with (key, value) pairs 
                relevant to this latent sampler
        '''
        return np.hstack((obs, state_infos[self.latent_name]))

    def _build(self):
        with tf.variable_scope(self.name):
            self.latent = tf.placeholder(tf.float32, shape=(None, self.dim), name=self.latent_name)

    def __getstate__(self):
        return dict(
            name=self.name, 
            dim=self.dim,
            latent_name=self.latent_name
        )

    def __setstate__(self, d):
        self.name = d['name']
        self.dim = d['dim']
        self.latent_name = d['latent_name']
        self._build()

def _categorical_latent_variable(dim, n_samples, pvals=None):
    pvals = np.ones(dim) / dim if pvals is None else pvals
    return np.random.multinomial(1, pvals, size=n_samples)

def _gaussian_latent_variable(dim, n_samples):
    return np.random.multivariate_normal(
        mean=np.zeros(dim),
        cov=np.eye(dim),
        size=n_samples
    )

def _build_latent_variable_function(variable_type):
    '''
    Factory method used because variable_type is used in multiple locations,
    and it is easier to pass around the string than it is to pass around one
    of these methods and check for type infomation each time
    '''
    if variable_type == 'categorical':
        return _categorical_latent_variable
    elif variable_type == 'gaussian':
        return _gaussian_latent_variable
    else:
        raise ValueError('variable_type not implemented: {}'.format(variable_type))

class UniformlyRandomLatentSampler(LatentSampler):

    def __init__(
            self, 
            scheduler,
            variable_type='categorical', 
            **kwargs):
        super(UniformlyRandomLatentSampler, self).__init__(**kwargs)
        self.scheduler = scheduler
        self.variable_type = variable_type
        self.n_samples = None
        self._latent_variable_function = _build_latent_variable_function(variable_type)
        
    def _update_latent_variables(self, observations):
        '''
        Updates latent variables based on what the scheduler says.

        Args:
            - observations: numpy array of shape (?, obs_dim)
        '''
        indicators = self.scheduler.should_update(observations)
        if any(indicators):
            new_latent = self._latent_variable_function(
                dim=self.dim, n_samples=self.n_samples)
            for (i, indicator) in enumerate(indicators[:self.n_samples]):
                if indicator:
                    self.latent_values[i] = new_latent[i]

    def encode(self, observations):
        '''
        For the case where the observations are available before hand, for example in 
        the supervised case, this function allows for iterating the latent sampler to
        get the latent values at each timestep. This is essentially performing inference
        / recognition / encoding, so it's named encode to be symmetric with encoders.

        Args:
            - observations: shape (n_samples, timesteps, input_dim) array
        '''
        n_samples, timesteps, _ = observations.shape
        self.reset([True] * n_samples)
        latents = np.zeros((n_samples, timesteps, self.dim))
        for t in range(timesteps):
            latents[:,t,:], _ = self.get_actions(observations[:,t])
        return latents

    def get_action(self, observation):
        '''
        Returns latent variable associated with current timestep and obs.

        Args:
            - observation: numpy array of shape (1, obs_dim)
        '''
        self._update_latent_variables(observation)
        return copy.deepcopy(self.latent_values[0]), dict(latent=copy.deepcopy(self.latent_values[0]))

    def get_actions(self, observations):
        '''
        Returns latent variables for current timestep and observations.

        Args:
            - observations: numpy array of shape (num_envs, obs_dim)
        '''
        self._update_latent_variables(observations)
        assert len(observations) == len(self.latent_values)
        return copy.deepcopy(self.latent_values), dict(latent=copy.deepcopy(self.latent_values))

    def reset(self, dones=None):
        '''
        resamples latent variables for the envionments which have just 
        completed an episode (dones[i] == True -> resample var i)

        Args:
            - dones: list of bools indicating whether the corresponding 
                environment has recently reached a terminal state
        '''
        dones = [True] if dones is None else dones
        if self.n_samples is None or len(dones) != self.n_samples:
            self.n_samples = len(dones)
            self.latent_values = self._latent_variable_function(
                dim=self.dim, n_samples=self.n_samples)

        self.scheduler.reset(dones)

    def __getstate__(self):
        d = super(UniformlyRandomLatentSampler, self).__getstate__()
        d['scheduler'] = self.scheduler
        d['variable_type'] = self.variable_type
        return d

    def __setstate__(self, d):
        super(UniformlyRandomLatentSampler, self).__setstate__(d)
        self.scheduler = d['scheduler']
        self.variable_type = d['variable_type']
        self.n_samples = None
        self._latent_variable_function = _build_latent_variable_function(self.variable_type)

