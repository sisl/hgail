
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
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
        dataset = RecognitionDataset(
            batch_size=1000,
            domain=True
        )

        # need a case where if you have domain adversarial training
        # the classifier isn't able to work, but if you don't have it
        # then it does work 
        n_samples = 100
        obs_dim = act_dim = 2
        xs = np.ones((n_samples, obs_dim))
        ys = np.zeros((n_samples, 2))
        ys[:,0] = 1
        xt = -np.ones((n_samples, obs_dim))
        yt = np.zeros((n_samples, 2))
        yt[:,1] = 1
        x = np.concatenate((xs,xt),0)
        y = np.concatenate((ys,yt),0)

        # random permute beforehand because otherwise it seems to have some 
        # unusual behavior because each batch contains only one of the domains
        # that is, the loss just keeps increasing with feature values also 
        # increasing
        # shouldn't the feature values just match over time?
        # instead seemingly arbitrary values just grow increasingly large
        # maybe it requires an l2 penalty to work? or dropout?
        idxs = np.random.permutation(n_samples * 2)
        data = dict(
            observations=x[idxs], 
            actions=x[idxs], 
            agent_infos=dict(latent=y[idxs]),
            env_infos=dict(domain=y[idxs])
        )

        with tf.Session() as session:
            latent_classifier = ObservationActionMLP(
                name='encoder', 
                hidden_layer_dims=[16,4],
                output_dim=2,
                return_features=True,
                dropout_keep_prob=1.,
                l2_reg=0.
            )

            domain_classifier = Classifier(
                name='domain_classifier',
                hidden_layer_dims=[16,16],
                output_dim=2,
                dropout_keep_prob=1.
            )

            recog = DomainAdvRecognitionModel(
                latent_classifier=latent_classifier,
                domain_classifier=domain_classifier,
                obs_dim=obs_dim,
                act_dim=act_dim,
                dataset=dataset, 
                variable_type='categorical',
                latent_dim=2,
                lambda_final=1e10,
                lambda_initial=1e10,
                grad_clip=1000.0,
                grad_scale=50.0,
                verbose=0
            )
            session.run(tf.global_variables_initializer())

            n_epochs = 500
            for epoch in range(n_epochs):
                recog.train(epoch, data)

            feed = {
                recog.x:x,
                recog.a:x,
                recog.c:y,
                recog.d:y
            }
            outputs_list = [
                recog.features, 
                recog.acc, 
                recog.domain_acc, 
                recog.domain_probs, 
                recog.probs,
                recog.gradients]
            features, acc, domain_acc, domain_probs, probs, grads = session.run(outputs_list, feed_dict=feed)
            src_features = features[:n_samples]
            tgt_features = features[n_samples:]

            self.assertTrue(np.abs(domain_probs[0][0] - .5) < .1)
            self.assertTrue(np.abs(domain_probs[n_samples][0] - .5) < .1)

if __name__ == '__main__':
    unittest.main()