
import gym
import joblib
import numpy as np
import os
import tensorflow as tf

from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
import rllab.misc.logger as logger
from rllab.envs.normalized_env import normalize

from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
from sandbox.rocky.tf.algos.trpo import TRPO

from hgail.critic.critic import WassersteinCritic
from hgail.misc.datasets import CriticDataset, RecognitionDataset
from hgail.envs.envs import TwoRoundNondeterministicRewardEnv
from hgail.policies.categorical_latent_var_mlp_policy import CategoricalLatentVarMLPPolicy
from hgail.policies.categorical_latent_sampler import CategoricalLatentSampler
from hgail.algos.gail import GAIL
from hgail.algos.hgail_impl import HGAIL
from hgail.samplers.hierarchy_sampler import HierarchySampler
from hgail.policies.latent_sampler import UniformlyRandomLatentSampler
from hgail.core.models import CriticNetwork, ObservationActionMLP
from hgail.recognition.recognition_model import RecognitionModel
from hgail.policies.scheduling import ConstantIntervalScheduler
from hgail.misc.utils import RewardHandler
import hgail.misc.utils

import simulate
import utils

# constants
use_replay_memory = True
use_critic_replay_memory = True
latent_dim_0 = 2
scheduler_k_0 = 10
latent_dim_1 = 3
scheduler_k_1 = 40
real_data_maxsize = None
batch_size = 8000
n_critic_train_epochs = 50
max_path_length = 1000
n_itr = 1000
critc_lr = .0001
level_1_step_size = .01
level_0_step_size = .01
use_env_rewards_1 = True # False
use_env_rewards_0 = False # False
critic_scale_1 = 0 # 1.
critic_scale_0 = 0.

# setup
env_id = "CartPole-v0"
exp_name = "CartPole-v0_2017_08_10_11_44"
phase = 'imitate_hgail'
exp_dir = utils.set_up_experiment(exp_name=exp_name, phase=phase, snapshot_gap=5)

# load env from the training process
snapshot_filepath = utils.latest_snapshot(exp_dir, phase='train')
env = utils.load_env(snapshot_filepath)
policy_filepath = '../data/experiments/{}/imitate/log/itr_500.pkl'.format(exp_name)
policy_param_values = utils.load_policy_param_values(policy_filepath)
# policy_param_values = None

# load critic dataset
expert_data_filepath = os.path.join(exp_dir, 'collection', 'expert_traj.h5')
data = hgail.misc.utils.load_dataset(expert_data_filepath, maxsize=real_data_maxsize)
data['actions'] = hgail.misc.utils.to_onehot(data['actions'])

if use_critic_replay_memory:
    critic_replay_memory = hgail.misc.utils.KeyValueReplayMemory(maxsize=4 *  batch_size)
else:
    critic_replay_memory = None

critic_dataset = CriticDataset(data, batch_size=4000, replay_memory=critic_replay_memory)

# session for actual training
with tf.Session() as session:
 
    # summary writer 
    summary_writer = tf.summary.FileWriter(
        os.path.join(exp_dir, phase, 'summaries'))

    # build the critic
    with tf.variable_scope('critic'):
        critic_network = ObservationActionMLP(
            name='critic', 
            hidden_layer_dims=[64,64]
        )
        critic = WassersteinCritic(
            obs_dim=env.observation_space.flat_dim,
            act_dim=env.action_space.n,
            dataset=critic_dataset, 
            network=critic_network,
            gradient_penalty=10.,
            optimizer=tf.train.RMSPropOptimizer(critc_lr),
            n_train_epochs=n_critic_train_epochs,
            summary_writer=summary_writer,
            verbose=2,
        )

    # level 2
    base_latent_sampler = UniformlyRandomLatentSampler(
        name='base_latent_sampler',
        dim=latent_dim_1,
        scheduler=ConstantIntervalScheduler(k=scheduler_k_1)
    )

    # level 1
    with tf.variable_scope('level_1'):
        recog_dataset_1 = RecognitionDataset(batch_size)
        recog_network_1 = ObservationActionMLP(
            name='recog_1', 
            hidden_layer_dims=[32,32],
            output_dim=latent_dim_1
        )
        recog_1 = RecognitionModel(
                    obs_dim=env.observation_space.flat_dim,
                    act_dim=env.action_space.n,
                    dataset=recog_dataset_1, 
                    network=recog_network_1,
                    variable_type='categorical',
                    latent_dim=latent_dim_1,
                    name='recognition_1',
                    verbose=2,
                    summary_writer=summary_writer
        )

        latent_sampler = CategoricalLatentSampler(
            scheduler=ConstantIntervalScheduler(k=scheduler_k_0),
            name='latent_sampler',
            policy_name='latent_sampler_policy',
            dim=latent_dim_0,
            env_spec=env.spec,
            latent_sampler=base_latent_sampler,
            max_n_envs=20
        )
        baseline_1 = LinearFeatureBaseline(env_spec=env.spec)

        algo_1 = TRPO(
            env=env,
            policy=latent_sampler,
            baseline=baseline_1,
            batch_size=batch_size,
            max_path_length=max_path_length,
            n_itr=n_itr,
            discount=0.99,
            step_size=level_1_step_size,
            sampler_cls=HierarchySampler,

        )
        reward_handler_1 = RewardHandler(use_env_rewards=use_env_rewards_1, critic_scale=critic_scale_1)
        level_1 = dict(
            algo=algo_1,
            reward_handler=reward_handler_1,
            recognition=recog_1,
            start_itr=0,
            end_itr=np.inf
        )

    # level 0 
    with tf.variable_scope('level_0'):
        recog_dataset_0 = RecognitionDataset(batch_size)
        recog_network_0 = ObservationActionMLP(
            name='recog_0', 
            hidden_layer_dims=[32,32],
            output_dim=latent_dim_0
        )
        recog_0 = RecognitionModel(
                    obs_dim=env.observation_space.flat_dim,
                    act_dim=env.action_space.n,
                    dataset=recog_dataset_0, 
                    network=recog_network_0,
                    variable_type='categorical',
                    latent_dim=latent_dim_0,
                    name='recognition_0',
                    verbose=2,
                    summary_writer=summary_writer
        )

        policy = CategoricalLatentVarMLPPolicy(
            policy_name="worker_policy",
            latent_sampler=latent_sampler,
            env_spec=env.spec,
            hidden_sizes=(32,32),
        )
        baseline_0 = LinearFeatureBaseline(env_spec=env.spec)

        algo_0 = TRPO(
            env=env,
            policy=policy,
            baseline=baseline_0,
            batch_size=batch_size,
            max_path_length=max_path_length,
            n_itr=n_itr,
            discount=0.99,
            step_size=level_0_step_size,
            sampler_args=dict(n_envs=1)
        )

        reward_handler_0 = RewardHandler(use_env_rewards=use_env_rewards_0, critic_scale=critic_scale_0)
        level_0 = dict(
            algo=algo_0,
            reward_handler=reward_handler_0,
            recognition=recog_0,
            start_itr=0,
            end_itr=0
        )

    # set the policy parameters if some have been loaded
    session.run(tf.global_variables_initializer())
    if policy_param_values is not None:
        policy.set_param_values(policy_param_values)
    
    # build hierarchy
    hierarchy = [level_0, level_1]
    if use_replay_memory:
        replay_memory = hgail.misc.utils.ReplayMemory(maxsize=1000)
    else:
        replay_memory = None
    algo = HGAIL(
        critic=critic,
        hierarchy=hierarchy,
        replay_memory=replay_memory
    )

    # run training
    algo.train(sess=session)
