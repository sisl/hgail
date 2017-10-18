
import collections
import h5py
import numpy as np
import sys

class Trajectory(collections.defaultdict):

    def __init__(self):
        super(Trajectory, self).__init__(list)

    def add(self, x, a, r, a_info):
        self['observations'].append(x)
        if type(a) == int or type(a) == float:
            a = [a]
        self['actions'].append(a)
        self['rewards'].append(r)
        self['agent_infos'].append(a_info)

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
        traj.add(x, a, r, a_info)
        if done: break
        x = nx
    return traj

def collect_trajectories(n_traj, env, policy, max_steps):
    trajs = []
    for traj in range(n_traj):
        sys.stdout.write('\rtraj: {} / {}'.format(traj + 1, n_traj))
        trajs.append(simulate(env, policy, max_steps))
    print('\ntrajectory collection complete')
    return trajs

