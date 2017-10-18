
import numpy as np
import tensorflow as tf
import unittest

from hgail.core.models import ConvPredictor

class TestConvPredictor(unittest.TestCase):

    def setUp(self):
        # reset graph before each test case
        tf.set_random_seed(1)
        np.random.seed(1)
        tf.reset_default_graph()    

    def test_forward(self):

        with tf.Session() as session:
            network = ConvPredictor(name='test')

            inputs = tf.placeholder(shape=(None, 64, 64, 1), dtype=tf.float32)
            network(inputs)
            session.run(tf.global_variables_initializer())
            x = np.zeros((2,64,64,1), dtype=np.float32)
            scores = network.forward(x)
            self.assertEqual(scores.shape, (2,1))

if __name__ == '__main__':
    unittest.main()