
import numpy as np

from rllab.misc.overrides import overrides
from sandbox.rocky.tf.algos.batch_polopt import BatchPolopt

import hgail.misc.utils

class HGAIL(BatchPolopt):

    def __init__(
            self,
            critic, 
            hierarchy,
            saver=None,
            saver_filepath=None,
            validator=None,
            snapshot_env=True,):
        self.critic = critic
        self.hierarchy = hierarchy
        self.saver = saver
        self.saver_filepath = saver_filepath
        self.validator = validator
        self.snapshot_env = snapshot_env

    @overrides
    def optimize_policy(self, itr, samples_data):
        """
        Update the critic and recognition model in addition to the policy
        
        Args:
            itr: iteration counter
            samples_data: dict of dict with level numbers as keys
                each sub dictionary resulting from process_samples
                keys: 'rewards', 'observations', 'agent_infos', 'env_infos', 'returns', 
                      'actions', 'advantages', 'paths'
                the values in the infos dicts can be accessed for example as:

                    samples_data['agent_infos']['prob']
    
                and the returned value will be an array of shape (batch_size, prob_dim)
        """
        self.critic.train(itr, samples_data[0])
        for (i, level) in enumerate(self.hierarchy):
            if level.get('start_itr', 0) <= itr and itr < level.get('end_itr', np.inf):
                level['algo'].optimize_policy(itr, samples_data[i])
                level['recognition'].train(itr, samples_data[i])

    @overrides
    def process_samples(self, itr, paths):
        """
        Augment path rewards with critic and recognition model rewards, doing so 
            separately for each level in the hierarchy
        
        Args:
            itr: iteration counter
            paths: list of dictionaries 
                each containing info for a single trajectory
                each with keys 'observations', 'actions', 'agent_infos', 'env_infos', 'rewards'

        Returns:
            - dict of dict, with topmost keys being the level numbers and lower keys
                being those usually included in samples_data
        """
        samples_data = dict()
        critic_rewards = self.critic.critique(itr, paths)
        for (i, level) in enumerate(self.hierarchy):
            if level.get('start_itr', 0) <= itr:
                recognition_rewards = level['recognition'].recognize(itr, paths, depth=i)
                level_paths = level['reward_handler'].merge(
                    paths, critic_rewards, recognition_rewards)
                samples_data[i] = level['algo'].process_samples(itr, level_paths)

        return samples_data

    @overrides
    def obtain_samples(self, itr):
        return self.hierarchy[0]['algo'].obtain_samples(itr)

    def _save(self, itr):
        """
        Save a tf checkpoint of the session.
        """
        # using keep_checkpoint_every_n_hours as proxy for iterations between saves
        if self.saver and (itr + 1) % self.saver._keep_checkpoint_every_n_hours == 0:

            # collect params (or stuff to keep in general)
            params = dict()
            params['critic'] = self.critic.network.get_param_values()

            # if the environment is wrapped in a normalizing env, save those stats
            normalized_env = hgail.misc.utils.extract_normalizing_env(self.env)
            if normalized_env is not None:
                params['normalzing'] = dict(
                    obs_mean=normalized_env._obs_mean,
                    obs_var=normalized_env._obs_var
                )

            # save hierarchy
            for i, level in enumerate(self.hierarchy):
                params[i] = dict()
                params[i]['policy'] = level['algo'].policy.get_param_values()
                
            # save params 
            save_dir = os.path.split(self.saver_filepath)[0]
            hgail.misc.utils.save_params(save_dir, params, itr, max_to_keep=50)

    def load(self, filepath):
        '''
        Load parameters from a filepath. Symmetric to _save. This is not ideal, 
        but it's easier than keeping track of everything separately.
        '''
        params = hgail.misc.utils.load_params(filepath)
        self.critic.network.set_param_values(params['critic'])
        normalized_env = hgail.misc.utils.extract_normalizing_env(self.env)
        if normalized_env is not None:
            normalized_env._obs_mean = params['normalzing']['obs_mean']
            normalized_env._obs_var = params['normalzing']['obs_var']

        # check for hierarchy
        initial = True
        for i, level in enumerate(self.hierarchy):
            if i in params.keys():
                initial = False
                level['algo'].policy.set_param_values(params[i]['policy'])

        # reaches this point without loading, we assume it is a save from 
        # the basic gail algorithm, that forms an initial save for hgail
        if initial:
            self.hierarchy[0]['algo'].policy.set_param_values(params['policy'])

    def _validate(self, itr, samples_data):
        """
        Run validation functions.
        """
        if self.validator:
            objs = dict(
                policy=self.policy, 
                critic=self.critic, 
                samples_data=samples_data,
                env=self.env)
            self.validator.validate(itr, objs)

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        self._save(itr)
        self._validate(itr, samples_data)
        return dict()

    @overrides
    def start_worker(self):
        self.hierarchy[0]['algo'].start_worker()

    @overrides
    def shutdown_worker(self):
        self.hierarchy[0]['algo'].shutdown_worker()

    @overrides
    def log_diagnostics(self, paths):
        self.hierarchy[0]['algo'].log_diagnostics(paths)

    def __getattr__(self, name):
        try:
            return getattr(self.hierarchy[0]['algo'], name)
        except Exception as e:
            print('class member with name {} requested'.format(name))
            print('class member not found in hgail')
            print('class member not found in first level algorithm in hierarhcy')
            raise(e)




