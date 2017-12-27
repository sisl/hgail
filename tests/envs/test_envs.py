
import copy
import numpy as np
import unittest

from hgail.envs.envs import TwoRoundContinuousDeterministicEnv

class TestTwoRoundContinuousDeterministicEnv(unittest.TestCase):

    def test_simple(self):
        env = TwoRoundContinuousDeterministicEnv()
        state = env.reset()
        x, x_des, t = state
        self.assertTrue(x > -1 and x < 1)
        self.assertTrue(x_des > -1 and x_des < 1)
        a = np.array([-.1])
        next_state, r, done, _ = env.step(a)
        nx, nx_des, n_t = next_state
        self.assertTrue(nx > -1 and nx < 1)
        self.assertTrue(nx_des > -1 and nx_des < 1)
        self.assertEqual(nx_des, x_des)
        self.assertTrue(nx == np.clip(x + a, -1, 1))
        self.assertEqual(r, -(nx - nx_des) ** 2)
        self.assertFalse(done)
        next_state, r, done, _ = env.step(a)
        nx, nx_des, t = next_state
        self.assertTrue(done)
        self.assertTrue(t == 2)
        
if __name__ == '__main__':
    unittest.main()