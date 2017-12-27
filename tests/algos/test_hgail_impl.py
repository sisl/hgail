'''
tests
1. test process_samples
2. test in env where it has to learn to make a decision at the beginning
3. test in env where it has to make multiple decisions throughout
those should both be simple
'''
import gym
import joblib
import numpy as np
import tensorflow as tf
import unittest

from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
import rllab.misc.logger as logger

from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
from sandbox.rocky.tf.algos.trpo import TRPO

from hgail.critic.critic import WassersteinCritic
from hgail.misc.datasets import CriticDataset, RecognitionDataset
from hgail.envs.envs import TwoRoundNondeterministicRewardEnv
from hgail.policies.categorical_latent_var_mlp_policy import CategoricalLatentVarMLPPolicy
from hgail.policies.categorical_latent_sampler import CategoricalLatentSampler
from hgail.algos.gail import GAIL
from hgail.algos.hgail_impl import HGAIL, Level
from hgail.samplers.hierarchy_sampler import HierarchySampler
from hgail.policies.latent_sampler import UniformlyRandomLatentSampler
from hgail.core.models import CriticNetwork, ObservationActionMLP
from hgail.recognition.recognition_model import RecognitionModel
from hgail.policies.scheduling import ConstantIntervalScheduler
from hgail.misc.utils import RewardHandler


def build_hgail(env, critic_dataset, batch_size):
        
    # critic
    with tf.variable_scope('critic'):
        critic_network = ObservationActionMLP(name='critic', hidden_layer_dims=[32,32])
        critic = WassersteinCritic(
            obs_dim=3,
            act_dim=2,
            dataset=critic_dataset, 
            network=critic_network,
            gradient_penalty=.01,
            optimizer=tf.train.AdamOptimizer(.001, beta1=.5, beta2=.9),
            n_train_epochs=50
        )


    # base latent variable sampler
    base_latent_sampler = UniformlyRandomLatentSampler(
        scheduler=ConstantIntervalScheduler(),
        name='base_latent_sampler',
        dim=3
    )

    with tf.variable_scope('level_1'):
        recog_dataset_1 = RecognitionDataset(batch_size=batch_size)
        recog_network_1 = ObservationActionMLP(
            name='recog', 
            hidden_layer_dims=[32,32],
            output_dim=3
        )
        recog_1 = RecognitionModel(
                    obs_dim=3,
                    act_dim=2,
                    dataset=recog_dataset_1, 
                    network=recog_network_1,
                    variable_type='categorical',
                    latent_dim=3,
                    name='recognition_1'
        )

        latent_sampler = CategoricalLatentSampler(
            scheduler=ConstantIntervalScheduler(k=1),
            name='latent_sampler',
            policy_name='latent_sampler_policy',
            dim=2,
            env_spec=env.spec,
            latent_sampler=base_latent_sampler,
            max_n_envs=20
        )
        baseline_1 = LinearFeatureBaseline(env_spec=env.spec)

        algo_1 = TRPO(
            env=env,
            policy=latent_sampler,
            baseline=baseline_1,
            batch_size=4000,
            max_path_length=100,
            n_itr=15,
            discount=0.99,
            step_size=0.01,
            sampler_cls=HierarchySampler,
        )
        reward_handler_1 = RewardHandler(use_env_rewards=False, critic_final_scale=1.)
        level_1 = Level(
            depth=1, 
            algo=algo_1, 
            reward_handler=reward_handler_1, 
            recognition_model=recog_1
        )

    with tf.variable_scope('level_0'):

        # recognition model
        recog_dataset_0 = RecognitionDataset(batch_size=batch_size)
        recog_network_0 = ObservationActionMLP(
            name='recog', 
            hidden_layer_dims=[32,32],
            output_dim=2
        )
        recog_0 = RecognitionModel(
                    obs_dim=3,
                    act_dim=2,
                    dataset=recog_dataset_0, 
                    network=recog_network_0,
                    variable_type='categorical',
                    latent_dim=2,
                    name='recognition_0'
        )

        policy = CategoricalLatentVarMLPPolicy(
            policy_name="policy",
            latent_sampler=latent_sampler,
            env_spec=env.spec
        )
        baseline_0 = LinearFeatureBaseline(env_spec=env.spec)

        algo_0 = TRPO(
            env=env,
            policy=policy,
            baseline=baseline_0,
            batch_size=4000,
            max_path_length=100,
            n_itr=5,
            discount=0.99,
            step_size=0.1,
            sampler_args=dict(n_envs=1)
        )

        reward_handler_0 = RewardHandler(use_env_rewards=False, critic_final_scale=1.)
        level_0 = Level(
            depth=0, 
            algo=algo_0, 
            reward_handler=reward_handler_0, 
            recognition_model=recog_0
        )

    hierarchy = [level_0, level_1]
    algo = HGAIL(
        critic=critic,
        hierarchy=hierarchy,
    )
    return algo

class TestHGAIL(unittest.TestCase):

    def setUp(self):
        # reset graph before each test case
        tf.set_random_seed(1)
        np.random.seed(1)
        tf.reset_default_graph()    

    def test_hgail_two_round_stochastic_env(self):
        
        env = TfEnv(TwoRoundNondeterministicRewardEnv())

        # dataset of one-hot obs and acts
        # optimal actions: 0, 1
        # first state
        n_expert_samples = 1000
        batch_size = 1000
        half = int(n_expert_samples / 2)
        rx = np.zeros((n_expert_samples, 3))
        rx[:half,2] = 1
        rx[half:,0] = 1
        ra = np.zeros((n_expert_samples, 2))
        ra[:half,0] = 1 
        ra[half:,1] = 1 
        
        critic_dataset = CriticDataset(dict(observations=rx, actions=ra), batch_size=batch_size)

        with tf.Session() as session:
            # build it
            algo = build_hgail(env, critic_dataset, batch_size)
            session.run(tf.global_variables_initializer())

            # run it!
            algo.train(sess=session)
            policy = algo.hierarchy[0].algo.policy

            # evaluate
            l0_state_infos = dict(latent=[[1,0]])
            l0_dist_2 = policy.dist_info([[0.,0.,1.]], l0_state_infos)['prob']
            l0_dist_0 = policy.dist_info([[1.,0.,0.]], l0_state_infos)['prob']

            l1_state_infos = dict(latent=[[0,1]])
            l1_dist_2 = policy.dist_info([[0.,0.,1.]], l1_state_infos)['prob']
            l1_dist_0 = policy.dist_info([[1.,0.,0.]], l1_state_infos)['prob']

            np.testing.assert_array_almost_equal(l0_dist_2, [[1,0]], 1)
            np.testing.assert_array_almost_equal(l0_dist_0, [[0,1]], 1)
            np.testing.assert_array_almost_equal(l1_dist_2, [[1,0]], 1)
            np.testing.assert_array_almost_equal(l1_dist_0, [[0,1]], 1)

if __name__ == '__main__':
    unittest.main()
