
import numpy as np
import sys
import unittest

from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.distributions.categorical import Categorical
from rllab.envs.grid_world_env import GridWorldEnv
from rllab.misc import tensor_utils
from sandbox.rocky.tf.samplers.vectorized_sampler import VectorizedSampler

from hgail.samplers.hierarchy_sampler import HierarchySampler

sys.path.append('..')
from utils import MockPolicy, MockDist, MockAlgo

def generate_depth_2_path(timesteps=4):
    observations = np.array([np.ones(1) * i for i in range(timesteps)])
    actions = np.array([np.ones(1) * i for i in range(timesteps)])
    rewards = np.array([np.ones(1) * i for i in range(timesteps)])

    agent_infos = dict(
                    prob=np.array([[1,0] for i in range(timesteps)]), # for actual action
                    latent=np.array([np.ones(1) * i * 2 for i in range(timesteps)]),
                    latent_info=dict(
                        prob=np.array([[0,1] for i in range(timesteps)]), # for latent 0
                        update=np.array([False for i in range(timesteps)]),
                        latent=np.array([np.ones(1) * i * 2 for i in range(timesteps)]),
                        latent_info=dict(
                            latent=np.array([np.ones(1) * i * 3 for i in range(timesteps)])
                        )
                    )
                )

    agent_infos['latent_info']['update'][0] = True
    agent_infos['latent_info']['update'][int(timesteps / 2)] = True

    path = dict(
            observations=observations,
            actions=actions,
            rewards=rewards,
            agent_infos=agent_infos,
            env_infos=dict()   
    )
    return path

class TestHierarchySampler(unittest.TestCase):

    def test_extract_depth(self):
        paths = [generate_depth_2_path(4), generate_depth_2_path(8)]
        algo = MockAlgo()
        sampler = HierarchySampler(depth=1, algo=algo)
        p0, p1 = sampler._extract_depth(paths)

        # check that the latent values were all moved down one depth
        np.testing.assert_array_equal(p0['observations'], [[0],[2]])
        np.testing.assert_array_equal(p0['actions'], [[0],[4]])
        np.testing.assert_array_equal(p0['rewards'], [[1],[5]])
        np.testing.assert_array_equal(p0['agent_infos']['prob'], [[0,1],[0,1]])
        np.testing.assert_array_equal(p0['agent_infos']['latent'], [[0],[6]])

        np.testing.assert_array_equal(p1['observations'], [[0],[4]])
        np.testing.assert_array_equal(p1['actions'], [[0],[8]])
        np.testing.assert_array_equal(p1['rewards'], [[6],[22]])
        np.testing.assert_array_equal(p1['agent_infos']['latent'], [[0],[12]])


    def test_extract_depth_discounting(self):
        # not sure what to do yet
        pass 
        # # with non-one discount factor
        # algo = MockAlgo(discount=.1)
        # sampler = hgail.hierarchy_sampler.HierarchySampler(depth=1, algo=algo)
        # p0, p1 = sampler._extract_depth(paths)

        # np.testing.assert_array_equal(p0['rewards'], [[.1],[5]])
        # np.testing.assert_array_equal(p1['rewards'], [[6],[22]])

    def test_process_samples(self):
        np.random.seed(1)
        env = GridWorldEnv()
        policy = MockPolicy(env.action_space)
        baseline = ZeroBaseline(env.spec)
        algo = MockAlgo(policy, env, baseline, discount=.5, batch_size=1)
        path_sampler = VectorizedSampler(algo=algo, n_envs=1)
        hierarchy_sampler = HierarchySampler(depth=1, algo=algo)
        path_sampler.start_worker()
        paths = path_sampler.obtain_samples(0)
        paths[0]['rewards'][0] = 1
        paths[0]['rewards'][-1] = 1
        paths = hierarchy_sampler.process_samples(0, paths)

        self.assertEqual(len(paths['observations']), 4)
        np.testing.assert_array_equal(paths['actions'], [[0],[2],[4],[6]])
        np.testing.assert_array_equal(paths['rewards'], [1.,0,0,.5])
        np.testing.assert_array_equal(paths['advantages'], [1.0625,.125,.25,.5])
        np.testing.assert_array_equal(paths['agent_infos']['latent'], [[0],[4],[8],[12]])
        np.testing.assert_array_equal(paths['agent_infos']['prob'], [[0,1],[0,1],[0,1],[0,1]])

if __name__ == '__main__':
    unittest.main()