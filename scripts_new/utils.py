
import datetime
import glob
import joblib
import numpy as np
import os
import tensorflow as tf

import rllab.misc.logger as logger

def maybe_mkdir(dirpath):
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)

def formatted_datetime():
    x = str(datetime.datetime.now().isoformat())
    x = x[:x.rfind(':')]
    x = x.replace('-','_')
    x = x.replace(':','_')
    x = x.replace('T','_')
    return x

def latest_snapshot(exp_dir, phase='train'):
    snapshot_dir = os.path.join(exp_dir, phase, 'log')
    snapshots = glob.glob('{}/itr_*.pkl'.format(snapshot_dir))
    latest = sorted(snapshots, reverse=True)[0]
    return latest

def load_env(filepath):
    with tf.Session() as session:
        d = joblib.load(filepath)
        env = d['env']
    tf.reset_default_graph()
    return env

def load_policy_param_values(filepath):
    with tf.Session() as session:
        d = joblib.load(filepath)
        policy = d['policy']
        values = policy.get_param_values()
    tf.reset_default_graph()
    return values

def write_args(args, filepath):
    np.save(filepath, args)

def set_up_experiment(
        exp_name, 
        phase, 
        exp_home='../data/experiments/',
        snapshot_gap=5):
    maybe_mkdir(exp_home)
    exp_dir = os.path.join(exp_home, exp_name)
    maybe_mkdir(exp_dir)
    phase_dir = os.path.join(exp_dir, phase)
    maybe_mkdir(phase_dir)
    log_dir = os.path.join(phase_dir, 'log')
    maybe_mkdir(log_dir)
    logger.set_snapshot_dir(log_dir)
    logger.set_snapshot_mode('gap')
    logger.set_snapshot_gap(snapshot_gap)
    return exp_dir

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
