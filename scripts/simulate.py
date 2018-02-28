
import argparse
import joblib
import os
import matplotlib.pyplot as plt
import numpy as np
import sys
import tensorflow as tf

from sandbox.rocky.tf.envs.vec_env_executor import VecEnvExecutor

import hgail.misc.simulation

import utils

COLORS = ['red', 'blue', 'green']
STATE_LABLES = ['Position', 'Velocity', 'Angle', 'Angular Velocity']

def collect(filepath, n_traj, max_steps):
    tf.reset_default_graph()
    with tf.Session() as session:
        d = joblib.load(filepath)
        policy = d['policy']
        env = d['env']
        trajectories = hgail.misc.simulation.collect_trajectories(n_traj, env, policy, max_steps)

    return trajectories

def collect_many(filepaths, n_traj, max_steps):
    trajs = []
    for filepath in filepaths:
        trajs += collect(filepath, n_traj, max_steps)
    return trajs

def visualize_latent(trajs, output_dir, maxtraj=1000):
    for depth in [0,1]:
        fig = plt.figure(figsize=(16,8))
        fig.suptitle("State Variables with Varying Skill Selection", fontsize=35)
        for traj in trajs[:maxtraj]:
            for i in range(4):
                plt.subplot(2,2,i+1)

                colors = []
                for j in range(len(traj['observations'])):
                    info = traj['agent_infos'][j]['latent_info']
                    for _ in range(depth):
                        info = info['latent_info']
                    colors.append(COLORS[np.argmax(info['latent'])])

                plt.scatter(
                    np.array(traj['observations'])[:,i], 
                    np.arange(len(traj['observations'])), 
                    c=colors,
                    s=1.
                )
                if i % 2 == 0:
                    plt.ylabel('Time', fontsize=30)
                plt.xlabel(STATE_LABLES[i], fontsize=30)

        output_filepath = os.path.join(output_dir, 'latent_{}'.format(depth))
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(output_filepath, transparent=False)
        plt.clf()

def load_env(filepath):
    env = joblib.load(filepath)
    return env

def load_env_policy(filepath):
    d = joblib.load(filepath)
    
    try:
        policy = d['policy']
        env = d['env']
    except:
        try:
            policy = d[0]['policy']
            env = d[0]['env']
        except Exception as e:
            print(d.keys())
            raise(e)

    return env, policy

def visualize(env, policy, n_traj, max_steps, render=False):
    trajs = []
    for i in range(n_traj):
        sys.stdout.write('\rtraj {} / {}'.format(i, n_traj))
        traj = hgail.misc.simulation.simulate(env, policy, max_steps, render=render)
        trajs.append(traj)
    print('\nmean path len: {}'.format(np.mean([len(d['rewards']) for d in trajs])))
    print('mean reward: {}'.format(np.mean([np.sum(d['rewards']) for d in trajs])))
    return trajs
        
if __name__ == '__main__':
    '''
    Comment out the first code block (starting with collect) to visualize the 
    learned infogail policy, comment out the second to collect expert trajectories
    '''
    parser = argparse.ArgumentParser()
    # logistics
    parser.add_argument('--exp_name', type=str, default='CartPole-v0')
    parser.add_argument('--itr', type=int, default=95)
    parser.add_argument('--mode', type=str, default='collect', help='one of collect, evaluate, or visualize')
    parser.add_argument('--n_traj', type=int, default=500, help='number of trajectories to collect or evaluate with')
    parser.add_argument('--max_steps', type=int, default=1000)

    args = parser.parse_args()

    exp_name = 'CartPole-v0'

    # collect expert trajectories
    if args.mode == 'collect':
        input_filepath = '../data/experiments/{}/train/log/itr_{}.pkl'.format(args.exp_name, args.itr)
        output_dir = '../data/experiments/{}/collection/'.format(args.exp_name)
        utils.maybe_mkdir(output_dir)
        output_filepath = os.path.join(output_dir, 'expert_traj.h5')
        trajectories = collect(input_filepath, n_traj=args.n_traj, max_steps=args.max_steps)
        hgail.misc.simulation.write_trajectories(trajectories, output_filepath)

    # evaluate 
    elif args.mode == 'evaluate' or args.mode == 'visualize':
        if args.mode == 'visualize':
            args.n_traj = 10
            render = True 
        else:
            render = False

        phase = 'imitate'
        input_filepath = '../data/experiments/{}/{}/log/itr_{}.pkl'.format(
            args.exp_name, phase, args.itr)
        with tf.Session() as session:
            env, policy = load_env_policy(input_filepath)
            trajs = visualize(
                env, 
                policy, 
                n_traj=args.n_traj, 
                max_steps=args.max_steps, 
                render=render
            )

