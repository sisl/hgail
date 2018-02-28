
import tensorflow as tf

from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
import rllab.misc.logger as logger

from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
from sandbox.rocky.tf.algos.trpo import TRPO

import utils

# constants
batch_size = 4000
max_path_length = 10000

env_id = "CartPole-v0"
exp_name = "CartPole-v0"
utils.set_up_experiment(exp_name=exp_name, phase='train')

env = GymEnv(env_id, force_reset=True, record_video=False)
env = normalize(env)
env = TfEnv(env)

policy = CategoricalMLPPolicy(
    name="policy",
    env_spec=env.spec,
    hidden_sizes=(32, 32)
)
baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=batch_size,
    max_path_length=max_path_length,
    n_itr=100,
    discount=0.999,
    step_size=0.01
)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    algo.train(sess=sess)
