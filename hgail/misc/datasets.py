
import copy
import numpy as np

from hgail.misc.utils import pad_tensor, compute_n_batches

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

class Dataset(object):

    def __init__(
            self,
            data,
            batch_size,
            action_normalizer=None,
            observation_normalizer=None,
            replay_memory=None,
            recurrent=False,
            flat_recurrent=False,
            use_random_scaling=False,
            random_scale_factor=.2,
            use_random_noise=False,
            random_noise_factor=.003):

        assert 'observations' in data.keys()
        assert 'actions' in data.keys() 

        assert not (flat_recurrent and recurrent)
        if recurrent: 
            assert 'valids' in data.keys()
            # expert data has already been padded to the max sequence length
            self.max_seq_len = data['observations'].shape[1]

        self.data = data
        self.batch_size = batch_size
        self.action_normalizer = action_normalizer
        self.observation_normalizer = observation_normalizer
        self.replay_memory = replay_memory
        self.recurrent = recurrent
        self.flat_recurrent = flat_recurrent
        self.use_random_scaling = use_random_scaling
        self.random_scale_factor = random_scale_factor
        self.use_random_noise = use_random_noise
        self.random_noise_factor = random_noise_factor
        self.n_samples = len(data['observations']) # number of real samples
        self.next_idx = 0

    def _normalize(self, data):
        # normalize actions in the dataset to ensure consistency with generated actions
        if self.action_normalizer:
            data['actions'] = self.action_normalizer(data['actions'])

        # typically obs will be normalized through an environment wrapper, but in some
        # cases it is more convenient to do it in the dataset
        if self.observation_normalizer:
            data['observations'] = self.observation_normalizer(data['observations'])

    def _apply_random_scale(self, x):
        '''
        randomly scales the time dimensions
        i.e., x is assumed to be shape (batch_size, timesteps, input_dim), and 
        the same rescale factor is applied across timesteps for a given input_dim 
        '''
        random = np.random.uniform(size=(x.shape[0], 1, x.shape[-1]))
        scales = (random - .5) * 2 * self.random_scale_factor + 1.
        x *= scales
        return x

    def _apply_random_scale_to_batch(self, batch):
        if self.use_random_scaling and self.recurrent:
            batch['rx'][...,:2] = self._apply_random_scale(batch['rx'][...,:2])
            batch['ra'][...,:2] = self._apply_random_scale(batch['ra'][...,:2])

        return batch

    def _apply_random_noise(self, x):
        noise = np.random.randn(*x.shape) * self.random_noise_factor
        x += noise
        return x

    def _apply_random_noise_to_batch(self, batch):
        if self.use_random_noise:
            batch['rx'][...,:2] = self._apply_random_noise(batch['rx'][...,:2])
            batch['ra'][...,:2] = self._apply_random_noise(batch['ra'][...,:2])

        return batch

    def _apply_randomness_to_batch(self, batch):
        batch = self._apply_random_scale_to_batch(batch)
        batch = self._apply_random_noise_to_batch(batch)
        return batch

    def _format(self, data):
        if self.recurrent:
            assert 'valids' in data.keys()
            # pad to max sequence length
            data['actions'] = pad_tensor(data['actions'], self.max_seq_len, axis=1)
            data['observations'] = pad_tensor(data['observations'], self.max_seq_len, axis=1)
            data['valids'] = pad_tensor(data['valids'], self.max_seq_len, axis=1)
        elif self.flat_recurrent:
            act_dim = data['actions'].shape[-1]
            data['actions'] = np.reshape(data['actions'], (-1, act_dim))
            obs_dim = data['observations'].shape[-1]
            data['observations'] = np.reshape(data['observations'], (-1, obs_dim))

    def batches(self, samples_data, store=True):
        raise NotImplementedError()

class CriticDataset(Dataset):

    def __init__(self, data, shuffle=True, **kwargs):
        super(CriticDataset, self).__init__(data, **kwargs)
        self.shuffle = shuffle
    
    def _shuffle(self):
        # optionally shuffle when wrapping around
        if self.shuffle:
            idxs = np.random.permutation(self.n_samples)
            self.data['observations'] = self.data['observations'][idxs]
            self.data['actions'] = self.data['actions'][idxs]
            if self.recurrent:
                self.data['valids'] = self.data['valids'][idxs]

    def batches(self, samples_data, store=True):
        
        assert 'observations' in samples_data.keys()
        assert 'actions' in samples_data.keys()

        # copy in order to avoid mutating data used elsewhere
        sd = copy.deepcopy(samples_data)

        # format incoming data if necessary
        self._format(sd)

        # normalize
        self._normalize(sd)

        # n_samples will determine the total number of samples on which to train
        n_samples = len(sd['observations'])

        # if using replay memory, store info from this samples_data
        # and then sample a batch from the previously stored data
        if self.replay_memory:
            keys = ['observations', 'actions']
            keys += ['valids'] if self.recurrent else []
            if store: self.replay_memory.add(keys, sd)
            sd = self.replay_memory.sample(keys, n_samples)
            
        # compute and yield batches
        n_batches = compute_n_batches(n_samples, self.batch_size)
        for bidx in range(n_batches):
            batch = dict()
            
            # batch of generated data
            gidxs, _ = select_batch_idxs(bidx * self.batch_size, self.batch_size, 0, n_samples)
            gx = sd['observations'][gidxs]
            ga = sd['actions'][gidxs]

            # batch of real data
            ridxs, self.next_idx = select_batch_idxs(self.next_idx, self.batch_size, 0, self.n_samples)
            rx = self.data['observations'][ridxs]
            ra = self.data['actions'][ridxs]

            # build batch 
            batch = dict(rx=rx, ra=ra, gx=gx, ga=ga)

            # valids if recurrent critic
            if self.recurrent:
                batch['g_valids'] = sd['valids'][gidxs]
                batch['r_valids'] = self.data['valids'][ridxs]

            # wrap around real data if reached the end of it
            if self.next_idx >= self.n_samples:
                # optionally shuffle when wrapping around
                self._shuffle()
                self.next_idx = 0

            # optional random scaling
            batch = self._apply_randomness_to_batch(batch)
            
            # yield a batch of data
            yield batch

class RecognitionDataset(object):
    
    def __init__(
            self,
            batch_size,
            action_normalizer=None,
            observation_normalizer=None,
            latent_variable_type='categorical',
            recurrent=False,
            flat_recurrent=False,
            conditional=False,
            domain=False,
            cond_key=None):
        self.batch_size = batch_size
        self.action_normalizer = action_normalizer
        self.observation_normalizer = observation_normalizer
        self.latent_variable_type = latent_variable_type
        self.recurrent = recurrent
        self.flat_recurrent = flat_recurrent
        assert cond_key is not None or not conditional
        self.conditional = conditional
        self.cond_key = cond_key
        self.domain = domain

    def _normalize(self, data):
        # normalize actions
        if self.action_normalizer:
            data['actions'] = self.action_normalizer(data['actions'])

        # normalize actions
        if self.observation_normalizer:
            data['observations'] = self.observation_normalizer(data['observations'])

    def _format(self, data):
        if self.flat_recurrent:
            act_dim = data['actions'].shape[-1]
            data['actions'] = np.reshape(data['actions'], (-1, act_dim))
            obs_dim = data['observations'].shape[-1]
            data['observations'] = np.reshape(data['observations'], (-1, obs_dim))
            latent_dim = data['agent_infos']['latent'].shape[-1]
            data['agent_infos']['latent'] = np.reshape(data['agent_infos']['latent'], (-1, latent_dim))

    def batches(self, samples_data):
    
        assert 'observations' in samples_data.keys()
        assert 'actions' in samples_data.keys()
        assert 'agent_infos' in samples_data.keys()
        assert 'latent' in samples_data['agent_infos'].keys()
        if self.conditional:
            assert self.cond_key in samples_data['agent_infos'].keys()
        if self.recurrent:
            assert 'valids' in samples_data.keys()

        # copy in order to avoid mutating data used elsewhere
        sd = copy.deepcopy(samples_data)

        # format incoming data if necessary
        self._format(sd)

        # optionally normalize
        self._normalize(sd)
                
        n_samples = len(sd['observations'])
        n_batches = compute_n_batches(n_samples, self.batch_size)
        for bidx in range(n_batches):
            
            # batch of generated data
            idxs, _ = select_batch_idxs(bidx * self.batch_size, self.batch_size, 0, n_samples)
            x = sd['observations'][idxs]
            a = sd['actions'][idxs]
            c = sd['agent_infos']['latent'][idxs]

            # concatenate x and conditional var if conditional
            if self.conditional:
                z = sd['agent_infos'][self.cond_key][idxs]
                x = np.concatenate((x, z), axis=-1)

            # formulate batch
            batch = dict(x=x, a=a, c=c)

            # add valids if recurrent
            if self.recurrent:
                batch['valids'] = sd['valids'][idxs]

            # if domain then extracts the domain from the env infos
            if self.domain:
                batch['d'] = sd['env_infos']['domain'][idxs]

            # yield a batch of data
            yield batch
