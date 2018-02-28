
import collections
import copy
import h5py
import numpy as np
import os
import tensorflow as tf

from rllab.envs.normalized_env import NormalizedEnv

from hgail.envs.vectorized_normalized_env import VectorizedNormalizedEnv

'''
Reward utils
'''

def batch_to_path_rewards(rewards, path_lengths):
    '''
    Args:
        - rewards: numpy array of shape (batch size, reward_dim)
        - path_lengths: list of lengths to be selected in groups from the row of rewards
    '''
    assert len(rewards) == sum(path_lengths)
    
    path_rewards = []
    s = 0
    for path_length in path_lengths:
        e = s + path_length
        path_rewards.append(rewards[s:e])
        s = e
    return path_rewards

def batch_timeseries_to_path_rewards(rewards, path_lengths):
    '''
    Converts dense array to list of lists each of corresponding len in path_lengths

    Args:
        - rewards: numpy array of shape (batch size, max sequence length, reward dim)
        - path_lengths: list of lengths to be selected from the rows of rewards
    '''
    assert len(rewards) == len(path_lengths)

    path_rewards = []
    for (i, path_length) in enumerate(path_lengths):
        path_rewards.append(rewards[i, :path_length])
    return path_rewards

class RewardHandler(object):
    
    def __init__(
            self, 
            use_env_rewards=True,
            critic_clip_low=-np.inf,
            critic_clip_high=np.inf,
            critic_initial_scale=1.,
            critic_final_scale=1.,
            recognition_initial_scale=1,
            recognition_final_scale=1.,
            augmentation_scale=1.,
            normalize_rewards=False,
            alpha=.01,
            max_epochs=10000,
            summary_writer=None):
        self.use_env_rewards = use_env_rewards
        self.critic_clip_low = critic_clip_low
        self.critic_clip_high = critic_clip_high

        self.critic_initial_scale = critic_initial_scale
        self.critic_final_scale = critic_final_scale
        self.critic_scale = critic_initial_scale
        
        self.recognition_initial_scale = recognition_initial_scale
        self.recognition_final_scale = recognition_final_scale
        self.recognition_scale = recognition_initial_scale

        self.augmentation_scale = augmentation_scale

        self.normalize_rewards = normalize_rewards
        self.alpha = alpha
        self.critic_reward_mean = 0.
        self.critic_reward_var = 1.
        self.recog_reward_mean = 0.
        self.recog_reward_var = 1.
        
        self.step = 0
        self.max_epochs = max_epochs
        self.summary_writer = summary_writer

    def _update_reward_estimate(self, rewards, reward_type):
        # unpack
        a = self.alpha
        mean = self.critic_reward_mean if reward_type == 'critic' else self.recog_reward_mean
        var = self.critic_reward_var if reward_type == 'critic' else self.recog_reward_var

        # update the reward mean using the mean of the rewards
        new_mean = (1 - a) * mean + a * np.mean(rewards)
        # update the variance with the mean of the individual variances
        new_var = (1 - a) * var + a * np.mean((rewards - mean) ** 2)

        # update class members
        if reward_type == 'critic':
            self.critic_reward_mean = new_mean
            self.critic_reward_var = new_var
        else:
            self.recog_reward_mean = new_mean
            self.recog_reward_var = new_var

    def _normalize_rewards(self, rewards, reward_type):
        self._update_reward_estimate(rewards, reward_type)
        var = self.critic_reward_var if reward_type == 'critic' else self.recog_reward_var
        return rewards / (np.sqrt(var) + 1e-8)

    def _update_scales(self):

        self.step += 1
        frac = np.minimum(self.step / self.max_epochs, 1)
        self.critic_scale = self.critic_initial_scale \
            + frac * (self.critic_final_scale - self.critic_initial_scale)
        self.recognition_scale = self.recognition_initial_scale \
            + frac * (self.recognition_final_scale - self.recognition_initial_scale)

    def merge(
            self, 
            paths, 
            critic_rewards=None, 
            recognition_rewards=None):
        """
        Add critic and recognition rewards to path rewards based on settings
        
        Args:
            paths: list of dictionaries as described in process_samples
            critic_rewards: list of numpy arrays of equal shape as corresponding path['rewards']
            recognition_rewards: same as critic rewards
        """
        # update relative reward scales
        self._update_scales()

        # combine the different rewards
        for (i, path) in enumerate(paths):

            shape = np.shape(path['rewards'])

            # env rewards
            if self.use_env_rewards:
                path['rewards'] = np.float32(path['rewards'])
            else:
                path['rewards'] = np.zeros(shape, dtype=np.float32)

            # critic rewards
            if critic_rewards is not None:
                critic_rewards[i] = np.clip(critic_rewards[i], self.critic_clip_low, self.critic_clip_high)
                if self.normalize_rewards:
                    critic_rewards[i] = self._normalize_rewards(
                        critic_rewards[i], reward_type='critic')
                path['rewards'] += self.critic_scale * np.reshape(critic_rewards[i], shape)

            # recognition rewards
            if recognition_rewards is not None:
                if self.normalize_rewards:
                    recognition_rewards[i] = self._normalize_rewards(
                        recognition_rewards[i], reward_type='recognition')
                path['rewards'] += self.recognition_scale * np.reshape(recognition_rewards[i], shape)

        # optionally write a summary
        self._log_merge()

        return paths

    def _log_merge(self):
        if self.summary_writer is not None:

            summary = tf.Summary(value=[
                tf.Summary.Value(tag="reward_handler/critic_reward_mean", simple_value=self.critic_reward_mean),
                tf.Summary.Value(tag="reward_handler/critic_reward_var", simple_value=self.critic_reward_var), 
                tf.Summary.Value(tag="reward_handler/recognition_reward_mean", simple_value=self.recog_reward_mean), 
                tf.Summary.Value(tag="reward_handler/recognition_reward_var", simple_value=self.recog_reward_var),
                tf.Summary.Value(tag="reward_handler/critic_scale", simple_value=self.critic_scale),
                tf.Summary.Value(tag="reward_handler/recognition_scale", simple_value=self.recognition_scale),
            ])
            self.summary_writer.add_summary(summary, self.step)
            self.summary_writer.flush()

'''
Data utils
'''

class ActionNormalizer(object):
    
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        
    def normalize(self, act):
        act = (act - self.mean) / self.std
        return act

    def unnormalize(self, act):
        act = act * self.std + self.mean
        return act
        
    def __call__(self, act):
        return self.normalize(act)

class ActionRangeNormalizer(object):
    '''
    Converts from [low,high] range to [-1,1] range
    It's the inverse of a normalizing wrapper around an environment
    This should be applied to real data, where low and high are the bounds 
    within the environment. The reason is that the agent actions should be
    output in the range [-1,1], then mapped to the actual ranges by the 
    environement wrapper `normalize`. Thus, we want the real data to go 
    through the inverse mapping. From the environment bounds to the [-1,1].
    '''
    
    def __init__(self, low, high):
        low = np.array(low)
        high = np.array(high)
        self.half_range = (high - low) / 2.
        self.mean = (high + low) / 2.
    
    def normalize(self, act):
        act = (act - self.mean) / self.half_range
        act = np.clip(act, -1, 1)
        return act

    def __call__(self, act):
        return self.normalize(act)

def load_dataset(filepath, maxsize=None):
    f = h5py.File(filepath, 'r')
    d = dict()
    for key in f.keys():
        if maxsize is None:
            d[key] = f[key].value
        else:
            d[key] = f[key].value[:maxsize]
    return d

def compute_n_batches(n_samples, batch_size):
    n_batches = int(n_samples / batch_size)
    if n_samples % batch_size != 0:
        n_batches += 1
    return n_batches

def select_batch_idxs(start_idx, batch_size, min_idx, max_idx):
    end_idx = start_idx + batch_size
    end_idx = min(end_idx, max_idx)
    idxs = np.arange(start_idx, end_idx, dtype=int)

    # if too few samples selected, then randomly select the rest from the full range
    if len(idxs) < batch_size:
        n_additional = batch_size - len(idxs)
        additional_idxs = np.random.randint(low=min_idx, high=max_idx, size=n_additional)
        idxs = np.hstack((idxs, additional_idxs))

    return idxs, end_idx

def save_params(output_dir, params, epoch, max_to_keep=None):
    # make sure output_dir exists
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # save 
    output_filepath = os.path.join(output_dir, 'itr_{}'.format(epoch))
    np.savez(output_filepath, params=params)

    # delete files if in excess of max_to_keep
    if max_to_keep is not None:
        files = [os.path.join(output_dir, f) 
                for f in os.listdir(output_dir) 
                if os.path.isfile(os.path.join(output_dir, f)) 
                and 'itr_' in f]
        sorted_files = sorted(files, key=os.path.getmtime, reverse=True)
        if len(sorted_files) > max_to_keep:
            for filepath in sorted_files[max_to_keep:]:
                os.remove(filepath)

def load_params(filepath):
    return np.load(filepath)['params'].item()

'''
numpy utils
'''

def to_onehot(values, dim=None):
    assert len(values.shape) == 2

    if dim is None:
        dim = np.max(values) + 1

    onehot = np.zeros((len(values), dim))
    onehot[np.arange(len(values)), values.reshape(-1)] = 1
    return onehot

def pad_tensor(x, max_len, axis):
    pad_widths = [(0,0) for _ in range(len(x.shape))]
    pad_widths[axis] = (0, max_len - x.shape[axis])
    return np.pad(x, (pad_widths), mode='constant')

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(logits, axis=-1):
    shape = logits.shape
    logits = logits.astype(np.float128).reshape(-1, shape[-1])
    x = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = x / np.sum(x, axis=-1, keepdims=True)
    invalid_idxs = np.where(np.sum(probs, axis=-1, keepdims=True) > 1.)[0]
    probs[invalid_idxs] -= (np.sum(probs[invalid_idxs], axis=-1, keepdims=True) - 1 + 1e-8) / probs.shape[-1]
    probs = probs.astype(np.float64)
    return probs.reshape(shape)

def multiple_pval_multinomial(probs):
    ps = []
    for prob in probs:
        p = np.random.multinomial(1, prob)
        ps.append(p)
    return np.array(ps)

def tile_concatenate(tiling_value, sequence):
    tiling = np.tile(np.expand_dims(tiling_value, 1), (1, sequence.shape[1], 1))
    sequence = np.concatenate((sequence, tiling), axis=-1)
    return sequence

def closest_factors(n):
    s = np.floor(np.sqrt(n))
    while n % s != 0:
        s -= 1
    return int(s), int(n // s)

def pad_stride_concat(a, window_left, stride=1):
    n_samples, input_dim = np.shape(a)
    padding = np.zeros((window_left, input_dim))
    a = np.concatenate((padding, a), axis=0)

    out = np.zeros((n_samples, (window_left + 1) * input_dim))
    for i in range(window_left, len(a)):
        out[i-window_left] = a[i-window_left:i+1].flatten()

    return out

def probabilistic_round(a):
    p = np.random.uniform(size=a.shape)
    frac = a % 1
    up = np.where(frac > p)
    down = np.where(frac <= p)
    a[up] = np.ceil(a[up])
    a[down] = np.floor(a[down])
    return a

def subselect_dict_list_idxs(d_l, key, idxs_l):
    for d, idxs in zip(d_l, idxs_l):
        sub_d = dict()
        for (k,v) in d[key].items():
            sub_d[k] = v[idxs]
        d[key] = sub_d

def flatten(arr):
    '''reshape to (-1, lastdim)'''
    return np.reshape(arr, (-1, np.shape(arr)[-1]))

'''
Replay Memory
'''

class ReplayMemory(object):

    def __init__(self, maxsize=None):
        self.maxsize = maxsize
        self.mem = []

    def add(self, paths):
        self.mem.extend(paths)
        if self.maxsize:
            self.mem = self.mem[-self.maxsize:]

    def sample(self, size):
        return np.random.choice(self.mem, size)

class KeyValueReplayMemory(object):

    def __init__(self, maxsize=None):
        self.maxsize = maxsize
        self.mem = collections.defaultdict(list)

    def add(self, keys, values):
        '''
        Adds keys from values to memory

        Args:
            - keys: the keys to add, list of hashable
            - values: dict containing each key in keys
        '''
        n_samples = len(values[keys[0]])
        for key in keys:
            assert len(values[key]) == n_samples, 'n_samples from each key must match'
            self.mem[key].extend(values[key])
            if self.maxsize:
                self.mem[key] = self.mem[key][-self.maxsize:]

    def sample(self, keys, size):
        '''
        Sample a batch of size for each key and return as a dict

        Args:
            - keys: list of keys
            - size: number of samples to select
        '''
        sample = dict()
        n_samples = len(self.mem[keys[0]])
        idxs = np.random.randint(0, n_samples, size)
        for key in keys:
            sample[key] = np.take(self.mem[key], idxs, axis=0)
        return sample

'''
rllab utils
'''
def extract_wrapped_env(env, typ):
    while not isinstance(env, typ):
        # descend to wrapped env
        if hasattr(env, 'wrapped_env'):
            env = env.wrapped_env
        # not the desired type, and has no wrapped env, return None
        else:
            return None
    # reaches this point, then the env is of the desired type, return it
    return env

def extract_normalizing_env(env):
    normalizing_env = extract_wrapped_env(env, NormalizedEnv)
    if normalizing_env is None:
        normalizing_env = extract_wrapped_env(env, VectorizedNormalizedEnv)
    return normalizing_env
