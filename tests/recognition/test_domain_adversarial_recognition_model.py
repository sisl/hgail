

import numpy as np
import tensorflow as tf
import unittest

from hgail.recognition.domain_adversarial_recognition_model import DomainAdvRecognitionModel
from hgail.misc.datasets import RecognitionDataset
from hgail.core.models import ObservationActionMLP, Classifier

class TestRecognitionModel(unittest.TestCase):

    def setUp(self):
        # reset graph before each test case
        tf.set_random_seed(1)
        np.random.seed(1)
        tf.reset_default_graph()  

    def test_train_simple(self):
        dataset = RecognitionDataset(
            batch_size=200,
            domain=True
        )

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
        d = np.zeros((n_samples, 2), dtype=np.int32)
        d[:(n_samples // 2), 0] = 1
        d[(n_samples // 2):, 1] = 1
        
        data = dict(
            observations=x, 
            actions=a, 
            agent_infos=dict(latent=c),
            env_infos=dict(domain=d))

        with tf.Session() as session:
            latent_classifier = ObservationActionMLP(
                name='encoder', 
                hidden_layer_dims=[32],
                output_dim=2,
                return_features=True
            )

            domain_classifier = Classifier(
                name='domain_classifier',
                hidden_layer_dims=[32],
                output_dim=2
            )

            recog = DomainAdvRecognitionModel(
                        latent_classifier=latent_classifier,
                        domain_classifier=domain_classifier,
                        obs_dim=1,
                        act_dim=1,
                        dataset=dataset, 
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

    def test_train_domain_matters(self):
        # what's a good test for this?
        pass
     
if __name__ == '__main__':
    unittest.main()