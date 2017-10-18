
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy

import numpy as np
import tensorflow as tf
import unittest

from hgail.policies.categorical_latent_var_mlp_policy import CategoricalLatentVarMLPPolicy
from hgail.envs.envs import TwoRoundNondeterministicRewardEnv

class TestGAILSerialization(unittest.TestCase):

    def setUp(self):
        # reset graph before each test case
        tf.set_random_seed(1)
        np.random.seed(1)
        tf.reset_default_graph()    

    def test_saver(self):
        savepath = 'data/model'
        env = TfEnv(TwoRoundNondeterministicRewardEnv())
        with tf.Session() as session:
            policy = CategoricalMLPPolicy(
                name="policy",
                env_spec=env.spec,
                hidden_sizes=[2,2]
            )
            session.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=100, keep_checkpoint_every_n_hours=.5)
            saver.save(session, savepath, global_step=0)
            params = policy.get_params()
            initial_values = policy.get_param_values()
            assign = tf.group(*[tf.assign(p, tf.zeros_like(p)) for p in params])
            session.run(assign)
            self.assertEqual(np.sum(policy.get_param_values()), 0)
            latest = tf.train.latest_checkpoint('data')
            saver.restore(session, latest)
            final_values = policy.get_param_values()
            np.testing.assert_array_equal(initial_values, final_values)
        

if __name__ == '__main__':
    unittest.main()