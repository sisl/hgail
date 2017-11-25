
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.zero_baseline import ZeroBaseline
from hgail.envs.new_env import DualGoalEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
import rllab.misc.logger as logger

from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
from sandbox.rocky.tf.algos.trpo import TRPO

import utils

# constants
batch_size = 5000
max_path_length = 100

exp_name = ["DualGoalEnv00","DualGoalEnv01","DualGoalEnv10","DualGoalEnv11"]
envL = [DualGoalEnv(task=0,typ=0),DualGoalEnv(task=0,typ=1),DualGoalEnv(task=1,typ=0),DualGoalEnv(task=1,typ=1)]

for i in range(4):
    env = envL[i]
    env = normalize(env)
    env = TfEnv(env)
    utils.set_up_experiment(exp_name=exp_name[i], phase='train')
    policy = CategoricalMLPPolicy(
        name=exp_name[i],
        env_spec=env.spec,
        hidden_sizes=(64, 32)
    )
    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=batch_size,
        max_path_length=max_path_length,
        n_itr=26,
        discount=0.99,
        step_size=0.015
    )
    algo.train()
