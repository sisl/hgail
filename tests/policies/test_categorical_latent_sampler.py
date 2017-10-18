
import copy
import numpy as np
import sys
import tensorflow as tf
import unittest

from sandbox.rocky.tf import spaces

from hgail.policies.categorical_latent_sampler import CategoricalLatentSampler
from hgail.policies.latent_sampler import UniformlyRandomLatentSampler
from hgail.policies.scheduling import ConstantIntervalScheduler

sys.path.append('..')
from utils import MockEnvSpec

def build_categorical_latent_sampler(
        base_dim=2, 
        base_scheduler_k=np.inf,
        dim=3,
        scheduler_k=np.inf):
    base_latent_sampler = UniformlyRandomLatentSampler(
            name='test_base',
            dim=base_dim,
            scheduler=ConstantIntervalScheduler(k=base_scheduler_k)
        )
    latent_sampler = CategoricalLatentSampler(
        name='test',
        policy_name='test',
        dim=dim,
        scheduler=ConstantIntervalScheduler(k=scheduler_k),
        env_spec=MockEnvSpec(
            action_space=spaces.Discrete(dim)
        ),
        latent_sampler=base_latent_sampler
    )
    return latent_sampler

class TestCategoricalLatentSamplerFunctions(unittest.TestCase):

    def setUp(self):
        # reset graph before each test case
        tf.set_random_seed(1)
        np.random.seed(1)
        tf.reset_default_graph()   

    def test_reset(self):
        latent_sampler = build_categorical_latent_sampler()
        try:
            latent_sampler.reset()
            latent_sampler.reset(dones=[True,True,True])
            latent_sampler.reset(dones=[True,False,True])
        except Exception as e:
            self.fail("resetting with different length dones fails with exception: {}".format(e))

    def test_get_action(self):
        with tf.Session() as session:
            latent_sampler = build_categorical_latent_sampler(
                scheduler_k=2
            )
            session.run(tf.global_variables_initializer())
            latent_sampler.reset()
            obs = [.5]
            action, agent_infos = latent_sampler.get_action(obs)
            self.assertEqual(action.shape, (3,))
            self.assertEqual(agent_infos['prob'].shape, (3,))
            self.assertTrue(agent_infos['update'][0])
            self.assertTrue('latent_info' in agent_infos.keys())
            latent_info = agent_infos['latent_info']
            action, agent_infos = latent_sampler.get_action(obs)
            self.assertFalse(agent_infos['update'][0])
            
    def test_get_actions(self):
        with tf.Session() as session:
            latent_sampler = build_categorical_latent_sampler(
                scheduler_k=2
            )
            session.run(tf.global_variables_initializer())
            latent_sampler.reset(dones=[True, True])
            obs = [[.5],[.5]]
            action, agent_infos = latent_sampler.get_actions(obs)
            self.assertEqual(action.shape, (2,3))
            self.assertEqual(agent_infos['prob'].shape, (2,3))
            self.assertTrue(agent_infos['update'][0])
            self.assertTrue(agent_infos['update'][1])
            self.assertTrue('latent_info' in agent_infos.keys())
            latent_info = agent_infos['latent_info']
            action, agent_infos = latent_sampler.get_actions(obs)
            self.assertFalse(agent_infos['update'][0])
            self.assertFalse(agent_infos['update'][1])
            action, agent_infos = latent_sampler.get_actions(obs)
            self.assertTrue(agent_infos['update'][0])
            self.assertTrue(agent_infos['update'][1])

class TestCategoricalLatentSampler(unittest.TestCase):

    def setUp(self):
        tf.set_random_seed(1)
        np.random.seed(1)
        tf.reset_default_graph() 

    def test_categorical_latent_sampler(self):
        with tf.Session() as session:
            # simple data
            n_samples = 100
            dim = 2
            x = np.random.rand(n_samples * 3).reshape(n_samples, 3)
            y = np.zeros((n_samples, 2))
            oneidxs = np.where(np.sum(x, axis=1) > 1.5)[0]
            zeroidxs = np.where(np.sum(x, axis=1) <= 1.5)[0]
            y[oneidxs] = [0,1]
            y[zeroidxs] = [1,0]

            latent_sampler = build_categorical_latent_sampler(dim=dim)
            probs = latent_sampler._output
            inputs = latent_sampler._inputs
            targets = tf.placeholder(shape=(None, 2), name='targets', dtype=tf.float32)
            loss = -tf.reduce_mean(tf.log(tf.reduce_sum(targets * probs, axis=1)))
            train_op = tf.train.AdamOptimizer(0.1).minimize(loss)

            session.run(tf.global_variables_initializer())

            n_epochs = 150
            for epoch in range(n_epochs):
                _, numloss = session.run([train_op, loss], 
                    feed_dict={targets:y, inputs:x})

            self.assertTrue(numloss < .01)




if __name__ == '__main__':
    unittest.main()