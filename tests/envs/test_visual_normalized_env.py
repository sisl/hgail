
import numpy as np
import sys
import unittest

from sandbox.rocky.tf import spaces

from hgail.envs.visual_normalized_env import VisualNormalizedEnv

sys.path.append('..')
from utils import MockEnv

class TestVisualNormalizedEnv(unittest.TestCase):

    def test_simple(self):
        observation_space = spaces.Box(np.zeros((180,120)), np.ones((180,120)) * 255.)
        action_space = spaces.Discrete(2)
        env = MockEnv(observation_space, action_space)
        size = 4
        env = VisualNormalizedEnv(env, size=(size,size), history_len=4)
        self.assertEqual(env.observation_space.low.shape, (size,size,4))
        x = env.reset()
        self.assertEqual(x.shape, (size,size,4))
        nx, r, t, _  = env.step(0)
        self.assertEqual(nx.shape, (size,size,4))
        np.testing.assert_array_equal(x[:,:,0], nx[:,:,1])
        env.step(0)
        env.step(0)
        nnx, r, t, _ = env.step(0)
        np.testing.assert_array_equal(nx[:,:,0], nnx[:,:,-1])

        
        
if __name__ == '__main__':
    unittest.main()