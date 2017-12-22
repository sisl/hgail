
import collections
import h5py
import numpy as np
import sys

from sandbox.rocky.tf.misc.tensor_utils import stack_tensor_dict_list, stack_tensor_list

class Trajectory(collections.defaultdict):

    def __init__(self):
        super(Trajectory, self).__init__(list)

    def add(self, x, a, r, a_info, env_info):
        self['observations'].append(x)
        if type(a) == int or type(a) == float:
            a = [a]
        self['actions'].append(a)
        self['rewards'].append(r)
        self['agent_infos'].append(a_info)
        self['env_infos'].append(env_info)

    def flatten(self):
        rtn = dict()
        rtn['observations'] = stack_tensor_list(self['observations'])
        rtn['actions'] = stack_tensor_list(self['actions'])
        rtn['rewards'] = stack_tensor_list(self['rewards'])
        # for each key, val in agent_infos and env_infos, stack them bring them to lowest level
        for k,v in stack_tensor_dict_list(self['agent_infos']).items():
            rtn[k] = v
        for k,v in stack_tensor_dict_list(self['env_infos']).items():
            rtn[k] = v
        return rtn

def write_trajectories(trajs, filepath, timeseries=False):
    '''
    just writes as (obs, act) pairs
    '''
    with h5py.File(filepath, 'w') as f:
        
        for key in ['observations', 'actions']:
            elements = []
            for traj in trajs:
                elements.extend(traj[key])
            elements = np.array(elements)
            f.create_dataset(key, data=elements)
            
        if timeseries:
            timesteps = []
            for traj in trajs:
                timesteps.append(len(traj['observations']))
            f.create_dataset('timesteps', data=timesteps)

def simulate(env, policy, max_steps, render=False):
    traj = Trajectory()
    x = env.reset()
    policy.reset()
    for step in range(max_steps):
        if render: env.render()
        a, a_info = policy.get_action(x)
        nx, r, done, e_info = env.step(a)
        traj.add(
            policy.observation_space.flatten(x), 
            a, 
            r, 
            a_info,
            e_info
        )
        if done: break
        x = nx
    return traj.flatten()

def collect_trajectories(n_traj, env, policy, max_steps):
    trajs = []
    for traj in range(n_traj):
        sys.stdout.write('\rtraj: {} / {}'.format(traj + 1, n_traj))
        trajs.append(simulate(env, policy, max_steps))
    print('\ntrajectory collection complete')
    return trajs

