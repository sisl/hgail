
import numpy as np

def vectorized_render_rollout(
        env, 
        policy, 
        max_path_length=np.inf, 
        deterministic=False,
        deterministic_key='mean'
    ):
    '''
    Description:
        Rollout function, but only for use with a vectorized environment and 
        policy, and for the purpose of rendering the environment, not collecting 
        trajectories or really anything else

    '''
    x = env.reset()
    n_envs = len(x)
    dones = np.array([True] * n_envs)
    env.render()
    ctr = 0
    while ctr < max_path_length:
        ctr += 1
        policy.reset(dones)
        a, a_info = policy.get_actions(x)
        if deterministic:
            if deterministic_key in a_info.keys():
                a = a_info[deterministic_key]
            else:
                raise ValueError('invalid key: {}. Valid keys: {}'.format(
                    deterministic_key, a_info.keys()))
        nx, r, dones, _ = env.step(a)
        if dones[0]:
            break
        x = nx
        env.render()



