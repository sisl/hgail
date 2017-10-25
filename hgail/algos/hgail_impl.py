
import numpy as np

from rllab.misc.overrides import overrides
from sandbox.rocky.tf.algos.batch_polopt import BatchPolopt

class HGAIL(BatchPolopt):

    def __init__(
            self,
            critic, 
            hierarchy,
            replay_memory=None):
        self.critic = critic
        self.hierarchy = hierarchy
        self.replay_memory = replay_memory

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
        if self.replay_memory:
            self.replay_memory.add(paths)
            paths = self.replay_memory.sample(len(paths))

        samples_data = dict()
        critic_rewards = self.critic.critique(itr, paths)
        for (i, level) in enumerate(self.hierarchy):
            if level.get('start_itr', 0) <= itr:
                recognition_rewards = level['recognition'].recognize(itr, paths, depth=i)
                level_paths = level['reward_handler'].merge(
                    paths, critic_rewards, recognition_rewards)
                samples_data[i] = level['algo'].process_samples(itr, paths)

        return samples_data

    @overrides
    def obtain_samples(self, itr):
        return self.hierarchy[0]['algo'].obtain_samples(itr)

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        snapshot = dict()
        algo_snapshots = {i:l['algo'].get_itr_snapshot(itr, samples_data) 
            for (i,l) in enumerate(self.hierarchy)}
        snapshot.update(algo_snapshots)
        
        return snapshot

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




