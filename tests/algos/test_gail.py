
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize

from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy

import gym
import joblib
import numpy as np
import os
import tensorflow as tf
import unittest

from hgail.critic.critic import WassersteinCritic
from hgail.misc.datasets import CriticDataset, RecognitionDataset
from hgail.envs.envs import TwoRoundNondeterministicRewardEnv
from hgail.policies.categorical_latent_var_mlp_policy import CategoricalLatentVarMLPPolicy
from hgail.algos.gail import GAIL
from hgail.policies.latent_sampler import UniformlyRandomLatentSampler
from hgail.core.models import CriticNetwork, ObservationActionMLP
from hgail.recognition.recognition_model import RecognitionModel
from hgail.policies.scheduling import ConstantIntervalScheduler
from hgail.misc.utils import RewardHandler

def train_gail(
        session, 
        env, 
        dataset,
        obs_dim=1,
        act_dim=2,
        n_itr=20,
        use_env_rewards=False,
        discount=.99,
        batch_size=4000,
        critic_scale=1.,
        gail_step_size=.01,
        critic_learning_rate=.001,
        policy_hid_layer_dims=[32,32],
        gradient_penalty=.1,
        critic_n_train_epochs=1,
        sampler_args=dict(),
        return_algo=False):

    network = CriticNetwork(hidden_layer_dims=[32,32])
    critic = WassersteinCritic(
        obs_dim=obs_dim,
        act_dim=act_dim,
        dataset=dataset, 
        network=network,
        verbose=2,
        gradient_penalty=gradient_penalty,
        optimizer=tf.train.AdamOptimizer(critic_learning_rate, beta1=.5, beta2=.9),
        n_train_epochs=critic_n_train_epochs
    )
    policy = CategoricalMLPPolicy(
        name="policy",
        env_spec=env.spec,
        hidden_sizes=policy_hid_layer_dims
    )
    baseline = LinearFeatureBaseline(env_spec=env.spec)

    reward_handler = RewardHandler(
        use_env_rewards=use_env_rewards,
        critic_final_scale=critic_scale)

    algo = GAIL(
        critic=critic,
        reward_handler=reward_handler,
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=batch_size,
        max_path_length=200,
        n_itr=n_itr,
        discount=discount,
        step_size=gail_step_size,
        sampler_args=sampler_args
    )
    session.run(tf.global_variables_initializer())

    if return_algo:
        return algo

    algo.train(sess=session)

    return policy, critic

class TestGAIL(unittest.TestCase):

    def setUp(self):
        # reset graph before each test case
        tf.set_random_seed(1)
        np.random.seed(1)
        tf.reset_default_graph()    

    def test_gail_one_round_deterministic_env(self):

        with tf.Session() as session:

            n_expert_samples = 1000
            rx = np.ones((n_expert_samples, 1))
            ra = np.zeros((n_expert_samples, 2))
            ra[:,1] = 1 # one hot actions
            dataset = CriticDataset(dict(observations=rx, actions=ra), batch_size=1000)

            env = TfEnv(GymEnv("OneRoundDeterministicReward-v0", force_reset=True))

            policy, critic = train_gail(session, env, dataset, use_env_rewards=False, n_itr=20)
            dist = policy.dist_info([[1.]])['prob']
            np.testing.assert_array_almost_equal(dist, [[0,1]], 2)

    def test_gail_two_round_deterministic_env(self):

        with tf.Session() as session:

            # dataset of one-hot obs and acts
            # optimal actions: 0, 1
            # first state
            n_expert_samples = 1000
            half = int(n_expert_samples / 2)
            rx = np.zeros((n_expert_samples, 3))
            rx[:half,2] = 1
            rx[half:,0] = 1
            ra = np.zeros((n_expert_samples, 2))
            ra[:half,0] = 1 
            ra[half:,1] = 1 
            dataset = CriticDataset(dict(observations=rx, actions=ra), batch_size=1000)

            env = TfEnv(GymEnv("TwoRoundDeterministicReward-v0", force_reset=True))

            policy, critic = train_gail(
                session, 
                env, 
                dataset,
                obs_dim=3,
                act_dim=2,
                use_env_rewards=False, 
                critic_scale=1.,
                n_itr=15,
                policy_hid_layer_dims=[32,32],
                batch_size=4000,
                critic_learning_rate=.001,
                gradient_penalty=1.,
                critic_n_train_epochs=10
            )
            dist_2 = policy.dist_info([[0.,0.,1.]])['prob']
            dist_0 = policy.dist_info([[1.,0.,0.]])['prob']
            np.testing.assert_array_almost_equal(dist_2, [[1,0]], 1)
            np.testing.assert_array_almost_equal(dist_0, [[0,1]], 1)

    def test_gail_two_round_stochastic_env(self):

        with tf.Session() as session:

            # dataset of one-hot obs and acts
            # optimal actions: 0, 1
            # first state
            n_expert_samples = 1000
            half = int(n_expert_samples / 2)
            rx = np.zeros((n_expert_samples, 3))
            rx[:half,2] = 1
            rx[half:,0] = 1
            ra = np.zeros((n_expert_samples, 2))
            ra[:half,0] = 1 
            ra[half:,1] = 1 
            dataset = CriticDataset(dict(observations=rx, actions=ra), batch_size=1000)

            env = TfEnv(TwoRoundNondeterministicRewardEnv())

            policy, critic = train_gail(
                session, 
                env, 
                dataset,
                obs_dim=3,
                act_dim=2,
                use_env_rewards=False, 
                critic_scale=1.,
                n_itr=15,
                policy_hid_layer_dims=[32,32],
                batch_size=4000,
                critic_learning_rate=.001,
                gradient_penalty=1.,
                critic_n_train_epochs=10,
                sampler_args=dict(n_envs=10)
            )
            dist_2 = policy.dist_info([[0.,0.,1.]])['prob']
            dist_0 = policy.dist_info([[1.,0.,0.]])['prob']
            np.testing.assert_array_almost_equal(dist_2, [[1,0]], 1)
            np.testing.assert_array_almost_equal(dist_0, [[0,1]], 1)

    def test_infogail_two_round_stochastic_env(self):

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
        
        with tf.Session() as session:
            # critic
            critic_dataset = CriticDataset(dict(observations=rx, actions=ra), batch_size=batch_size)
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

            # recognition model
            recog_dataset = RecognitionDataset(batch_size=batch_size)
            recog_network = ObservationActionMLP(
                name='recog', 
                hidden_layer_dims=[32,32],
                output_dim=2
            )
            recog = RecognitionModel(
                        obs_dim=3,
                        act_dim=2,
                        dataset=recog_dataset, 
                        network=recog_network,
                        variable_type='categorical',
                        latent_dim=2
            )

            # policy
            env.spec.num_envs = 10
            latent_sampler = UniformlyRandomLatentSampler(
                scheduler=ConstantIntervalScheduler(),
                name='latent_sampler',
                dim=2
            )
            policy = CategoricalLatentVarMLPPolicy(
                policy_name="policy",
                latent_sampler=latent_sampler,
                env_spec=env.spec
            )

            # gail
            reward_handler = RewardHandler(
                use_env_rewards=False,
                critic_final_scale=1.
            )
            baseline = LinearFeatureBaseline(env_spec=env.spec)
            algo = GAIL(
                critic=critic,
                recognition=recog,
                reward_handler=reward_handler,
                env=env,
                policy=policy,
                baseline=baseline,
                batch_size=4000,
                max_path_length=200,
                n_itr=15,
                discount=.99,
                step_size=.01,
                sampler_args=dict(n_envs=env.spec.num_envs)
            )

            session.run(tf.global_variables_initializer())

            # run it!
            algo.train(sess=session)

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