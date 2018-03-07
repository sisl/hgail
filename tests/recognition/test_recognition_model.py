
import numpy as np
import tensorflow as tf
import unittest

from hgail.recognition.recognition_model import RecognitionModel
from hgail.misc.datasets import RecognitionDataset
from hgail.core.models import ObservationActionMLP

class TestRecognitionModel(unittest.TestCase):

    def setUp(self):
        # reset graph before each test case
        tf.set_random_seed(1)
        np.random.seed(1)
        tf.reset_default_graph()  

    def test_recognize_categorical(self):

        network = ObservationActionMLP(
                name='recognition', 
                hidden_layer_dims=[16],
                output_dim=2)
        dataset = RecognitionDataset(batch_size=10)

        with tf.Session() as session:

            recog = RecognitionModel(
                        obs_dim=1,
                        act_dim=1,
                        dataset=dataset, 
                        network=network,
                        variable_type='categorical',
                        latent_dim=2
                    )
            session.run(tf.global_variables_initializer())
            paths = [
                dict(observations=[[1],[2]], actions=[[1],[2]], rewards=[[1],[2]], 
                    agent_infos=dict(latent_info=dict(latent=[[1,0],[0,1]]))),
                dict(observations=[[1],[2],[3]], actions=[[1],[2],[3]], rewards=[[1],[2],[3]],
                    agent_infos=dict(latent_info=dict(latent=[[1,0],[0,1],[1,0]]))),
                dict(observations=[[1]], actions=[[1]], rewards=[[1]],
                    agent_infos=dict(latent_info=dict(latent=[[1,0]]))),
            ]
            rewards = recog.recognize(1, paths)
            self.assertTrue(len(rewards[0]) == 2)
            self.assertTrue(len(rewards[1]) == 3)
            self.assertTrue(len(rewards[2]) == 1)  

    def test_train_categorical(self):
        dataset = RecognitionDataset(batch_size=200)

        # create dataset where x and a from unit gaussian and 
        # c = [1,0] if x+a < 0 else [0,1]
        n_samples = 2000
        x = np.random.randn(n_samples).reshape(-1, 1)
        a = np.random.randn(n_samples).reshape(-1, 1)
        c = np.zeros((n_samples, 2), dtype=np.int32)
        zero_idxs = np.where(x + a < 0)[0]
        one_idxs = np.where(x + a >= 0)[0]
        c[zero_idxs, 0] = 1
        c[one_idxs, 1] = 1
        c = np.int32(c)
        
        data = dict(observations=x, actions=a, agent_infos=dict(latent=c))

        with tf.Session() as session:
            network = ObservationActionMLP(
                name='recognition', 
                hidden_layer_dims=[16],
                output_dim=2)
            recog = RecognitionModel(
                        obs_dim=1,
                        act_dim=1,
                        dataset=dataset, 
                        network=network,
                        variable_type='categorical',
                        latent_dim=2
                    )
            session.run(tf.global_variables_initializer())

            n_epochs = 100
            for epoch in range(n_epochs):
                recog.train(epoch, data)

            probs = recog._probs(data['observations'], data['actions'])
            idxs = np.argmax(c, axis=1)
            loss = -np.log(probs[np.arange(n_samples), idxs]).mean()
            self.assertTrue(loss < .1)

    def test_recognize_gaussian(self):

        network = ObservationActionMLP(
                name='recognition', 
                hidden_layer_dims=[16],
                output_dim=2 * 2)
        dataset = RecognitionDataset(batch_size=10)

        with tf.Session() as session:

            recog = RecognitionModel(
                        obs_dim=1,
                        act_dim=1,
                        dataset=dataset, 
                        network=network,
                        variable_type='gaussian',
                        latent_dim=2
                    )
            session.run(tf.global_variables_initializer())
            paths = [
                dict(observations=[[1],[2]], actions=[[1],[2]], rewards=[[1],[2]], 
                    agent_infos=dict(latent_info=dict(latent=[[.5,.1],[.5,.1]]))),
                dict(observations=[[1],[2],[3]], actions=[[1],[2],[3]], rewards=[[1],[2],[3]],
                    agent_infos=dict(latent_info=dict(latent=[[-.1,.1],[.4,.4],[.6,.7]]))),
                dict(observations=[[1]], actions=[[1]], rewards=[[1]],
                    agent_infos=dict(latent_info=dict(latent=[[-1.,1.]]))),
            ]
            rewards = recog.recognize(1, paths)
            self.assertTrue(len(rewards[0]) == 2)
            self.assertTrue(len(rewards[1]) == 3)
            self.assertTrue(len(rewards[2]) == 1)  

    def test_train_gaussian(self):
        dataset = RecognitionDataset(batch_size=200)

        n_samples = 100
        latent_dim = 2
        x = np.random.randn(n_samples, latent_dim)
        a = np.random.randn(n_samples, latent_dim)
        mean = x + a
        sigma = np.exp(x * a / 2)
        c = []
        for (mean_i, sigma_i) in zip(mean, sigma):
            c.append(np.random.multivariate_normal(mean_i, np.eye(latent_dim)*sigma_i))
        c = np.array(c)
        
        data = dict(observations=x, actions=a, agent_infos=dict(latent=c))

        with tf.Session() as session:
            network = ObservationActionMLP(
                name='recognition', 
                hidden_layer_dims=[64,64],
                output_dim=latent_dim * 2)
            recog = RecognitionModel(
                        obs_dim=2,
                        act_dim=2,
                        dataset=dataset, 
                        network=network,
                        variable_type='gaussian',
                        latent_dim=latent_dim,
                        optimizer=tf.train.AdamOptimizer(.001, beta1=.5, beta2=.9),
                        verbose=0
                    )
            session.run(tf.global_variables_initializer())

            n_epochs = 200
            for epoch in range(n_epochs):
                recog.train(epoch, data)

            probs = recog._probs(data['observations'], data['actions'], data['agent_infos']['latent'])
            loss = np.mean(-np.log(probs))
            self.assertTrue(loss < 1.) # just whether it can overfit
     
if __name__ == '__main__':
    unittest.main()