import networkx as nx
import joblib
import os
import matplotlib.pyplot as plt
import numpy as np
import sys
import tensorflow as tf
from hgail.envs.new_env import DualGoalEnv
from sandbox.rocky.tf.envs.vec_env_executor import VecEnvExecutor
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.envs.base import TfEnv
import hgail.misc.simulation
from random import shuffle
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
        # sys.stdout.write('\rtraj {} / {}'.format(i, n_traj))
        traj = hgail.misc.simulation.simulate(env, policy, max_steps, render=render)
        trajs.append(traj)
    print('{}'.format(np.mean([len(d['rewards']) for d in trajs])))
    return trajs

def plot_trajectories(trajs,num):
    X=[]
    Y=[]
    dic={}
    for i in range(9):
        for j in range(9):
            dic[(i,j)]=0
    for i in range(num):
        X+=[map(lambda x:x[0][0],map(np.nonzero,trajs[i]['observations']))]
        Y+=[map(lambda x:x[0][1]-9,map(np.nonzero,trajs[i]['observations']))]
    for i in range(len(X)):
        for j in range(len(X[i])):
            dic[(X[i][j],Y[i][j])]+=1
    print '(0,4)',dic[(0,4)]
    print '(4,0)',dic[(4,0)]
    print '(8,4)',dic[(8,4)]
    # print dic[(5,5)]
    # for i in range(6):
    #     for j in range(6):
    #         print dic[(i,j)],
    #     print '\n'  

    # for i in range(len(X)):
    #     plt.plot(X[i],Y[i],'r')
    # plt.axis([-1,9,-1,9])
    # plt.show()

if __name__ == '__main__':
    '''
    Comment out the first code block (starting with collect) to visualize the 
    learned infogail policy, comment out the second to collect expert trajectories
    '''

    exp_list = ['DualGoalEnv00','DualGoalEnv10',"DualGoalEnv01","DualGoalEnv11"]

    # collect expert trajectories
    # itr = 25
    # trajectories=[]
    # for exp_name in exp_list:
    #     input_filepath = '../data/experiments/{}/train/log/itr_{}.pkl'.format(exp_name, itr)
    #     output_dir = '../data/experiments/{}/collection/'.format(exp_name)
    #     utils.maybe_mkdir(output_dir)
    #     output_filepath = os.path.join(output_dir, 'expert_traj.h5')
    #     trajectories+= collect(input_filepath, n_traj=100, max_steps=1000)
    #     # plot_trajectories(trajs)
    # shuffle(trajectories)
    # hgail.misc.simulation.write_trajectories(trajectories, output_filepath)

    # visualzie gail policy
    # itr = 215
    for itr in [215]:
        tf.reset_default_graph()
        phase = 'imitate'
        input_filepath = '../data/experiments/{}/{}/log/itr_{}.pkl'.format('DualGoalEnv11', phase, itr)
        with tf.Session() as session:
            env, policy = load_env_policy(input_filepath)
            env = DualGoalEnv(typ=2,task=1)
            env = normalize(env)
            env = TfEnv(env)
            trajs = visualize(env, policy, n_traj=100, max_steps=100, render=False)
            l1=[]
            l2=[]
            for i in range(100):
                if(list(trajs[i]['agent_infos'][0]['latent'])==[1,0]):
                    l1+=[trajs[i]]
                else:
                    l2+=[trajs[i]]
            plot_trajectories(l1,len(l1))
            plot_trajectories(l2,len(l2))

