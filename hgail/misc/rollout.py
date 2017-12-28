
import numpy as np

def vectorized_render_rollout(env, policy, max_path_length=np.inf):
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
        a, _ = policy.get_actions(x)
        nx, r, dones, _ = env.step(a)
        if dones[0]:
            break
        x = nx
        env.render()