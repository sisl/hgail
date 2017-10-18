
import numpy as np
import tensorflow as tf
import unittest

from hgail.critic.critic import WassersteinCritic
from hgail.misc.datasets import CriticDataset
from hgail.core.models import CriticNetwork

class TestWassersteinCritic(unittest.TestCase):

    def setUp(self):
        # reset graph before each test case
        tf.set_random_seed(1)
        np.random.seed(1)
        tf.reset_default_graph()    

    def test_critique(self):

        network = CriticNetwork(hidden_layer_dims=[2])
        rx = np.zeros((10,2))
        ra = np.zeros((10,1))
        dataset = CriticDataset(dict(observations=rx, actions=ra), batch_size=10)

        with tf.Session() as session:

            critic = WassersteinCritic(
                        obs_dim=1,
                        act_dim=1,
                        dataset=dataset, 
                        network=network
                    )
            session.run(tf.global_variables_initializer())
            paths = [
                dict(observations=[[1],[2]], actions=[[1],[2]], rewards=[[1],[2]]),
                dict(observations=[[1],[2],[3]], actions=[[1],[2],[3]], rewards=[[1],[2],[3]]),
                dict(observations=[[1]], actions=[[1]], rewards=[[1]]),
            ]
            rewards = critic.critique(1, paths)
            self.assertTrue(len(rewards[0]) == 2)
            self.assertTrue(len(rewards[1]) == 3)
            self.assertTrue(len(rewards[2]) == 1)

    def test_train(self):

        with tf.Session() as session:

            network = CriticNetwork(hidden_layer_dims=[24])
            batch_size = 10
            obs_dim = 2
            act_dim = 1
            real_data = dict(
                observations=np.ones((batch_size, obs_dim)) * .5, 
                actions=np.ones((batch_size, act_dim)) * .5)
            fake_data = dict(
                observations=np.ones((batch_size, obs_dim)) * -.5, 
                actions=np.ones((batch_size, act_dim)) * -.5)
            dataset = CriticDataset(real_data, batch_size=batch_size)

            critic = WassersteinCritic(
                        obs_dim=obs_dim,
                        act_dim=act_dim,
                        dataset=dataset, 
                        network=network,
                        gradient_penalty=.01
                    )

            session.run(tf.global_variables_initializer())

            n_epochs = 500
            for epoch in range(n_epochs):
                critic.train(epoch, fake_data)

            real_rewards = critic.network.forward(
                real_data['observations'], real_data['actions'])
            fake_rewards = critic.network.forward(
                fake_data['observations'], fake_data['actions'])

            self.assertTrue(real_rewards[0] > 1)
            self.assertTrue(fake_rewards[0] < -1)

if __name__ == '__main__':
    unittest.main()