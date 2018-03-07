
import copy
import numpy as np
import sys
import unittest

import hgail.policies.latent_sampler
from hgail.policies.latent_sampler import UniformlyRandomLatentSampler
from hgail.policies.scheduling import ConstantIntervalScheduler

sys.path.append('..')
from utils import MockEnvSpec

class TestLatentVariableFunctions(unittest.TestCase):

    def test_categorical_latent_variable(self):
        dim = 3
        n_samples = 100
        samples = hgail.policies.latent_sampler._categorical_latent_variable(dim, n_samples)
        self.assertTrue(np.shape(samples) == (100, 3))
        self.assertTrue(all(np.sum(samples, axis=1) == 1))

    def test_gaussian_latent_variable(self):
        dim = 3
        n_samples = 100
        samples = hgail.policies.latent_sampler._gaussian_latent_variable(dim, n_samples)
        self.assertTrue(np.shape(samples) == (100, 3))

class TestUniformlyRandomLatentSampler(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)  

    def test_reset(self):

        # single env
        dim = 3
        env_spec = MockEnvSpec()
        sampler = UniformlyRandomLatentSampler(
            scheduler=ConstantIntervalScheduler(), name='test', dim=dim)
        dones = [True]
        sampler.reset(dones)
        action, _ = sampler.get_action(None)
        self.assertTrue(sampler.latent_values.shape == (1,3))
        self.assertTrue(np.sum(sampler.latent_values, axis=1) == 1)

        # multi env
        env_spec = MockEnvSpec(num_envs=2)
        dim = 100
        sampler = UniformlyRandomLatentSampler(
            scheduler=ConstantIntervalScheduler(), name='test', dim=dim)
        dones = [True, True]
        sampler.reset(dones)

        self.assertTrue(sampler.latent_values.shape == (2,dim))

        actions_1, _ = sampler.get_actions([None] * 2)
        sampler.reset(dones)

        actions_2, _ = sampler.get_actions([None] * 2)
        self.assertEqual(sampler.latent_values.shape, (2, dim))
        self.assertNotEqual(tuple(np.argmax(actions_1, axis=1)), 
            tuple(np.argmax(actions_2, axis=1)))

        dones = [False, True]
        sampler.reset(dones)
        np.testing.assert_array_equal(np.sum(sampler.latent_values, axis=1), [1,1])

    def test_get_action(self):
        dim = 3
        env_spec = MockEnvSpec()
        sampler = UniformlyRandomLatentSampler(
            scheduler=ConstantIntervalScheduler(), name='test', dim=dim)
        sampler.reset([True])
        obs = [[0,1]]
        latent, agent_info = sampler.get_action(obs)
        self.assertTrue('latent' in agent_info.keys())

        sampler.reset([True])
        obs = [[0,0,1]]
        latent, agent_info = sampler.get_action(obs)
        self.assertEqual(latent.shape, (3,))
        self.assertEqual(sum(latent), 1)

    def test_get_actions(self):
        dim = 2
        env_spec = MockEnvSpec(num_envs=5)
        sampler = UniformlyRandomLatentSampler(
            scheduler=ConstantIntervalScheduler(),
            name='test', 
            dim=dim
        )
        sampler.reset([True] * 5)

        # scalar observations case
        obs = np.zeros((env_spec.num_envs, 3))
        latent, agent_info = sampler.get_actions(obs)
        self.assertEqual(latent.shape, (env_spec.num_envs, 2))

if __name__ == '__main__':
    unittest.main()