
import numpy as np
import os
import tensorflow as tf

from rllab.misc.overrides import overrides

from sandbox.rocky.tf.algos.trpo import TRPO

import hgail.misc.utils

class GAIL(TRPO):
    """
    Generative Adversarial Imitation Learning
    """
    def __init__(
            self,
            critic=None,
            recognition=None,
            reward_handler=hgail.misc.utils.RewardHandler(),
            saver=None,
            saver_filepath=None,
            validator=None,
            **kwargs):
        """
        Args:
            critic: 
            recognition:
        """
        self.critic = critic
        self.recognition = recognition
        self.reward_handler = reward_handler
        self.saver = saver
        self.saver_filepath = saver_filepath
        self.validator = validator
        super(GAIL, self).__init__(**kwargs)
    
    @overrides
    def optimize_policy(self, itr, samples_data):
        """
        Update the critic and recognition model in addition to the policy
        
        Args:
            itr: iteration counter
            samples_data: dictionary resulting from process_samples
                keys: 'rewards', 'observations', 'agent_infos', 'env_infos', 'returns', 
                      'actions', 'advantages', 'paths'
                the values in the infos dicts can be accessed for example as:

                    samples_data['agent_infos']['prob']
    
                and the returned value will be an array of shape (batch_size, prob_dim)
        """
        super(GAIL, self).optimize_policy(itr, samples_data) 
        if self.critic is not None:
            self.critic.train(itr, samples_data)
        if self.recognition is not None:
            self.recognition.train(itr, samples_data)
            
    @overrides
    def process_samples(self, itr, paths):
        """
        Augment path rewards with critic and recognition model rewards
        
        Args:
            itr: iteration counter
            paths: list of dictionaries 
                each containing info for a single trajectory
                each with keys 'observations', 'actions', 'agent_infos', 'env_infos', 'rewards'
        """
        # compute critic and recognition rewards and combine them with the path rewards
        critic_rewards = self.critic.critique(itr, paths) if self.critic else None
        recognition_rewards = self.recognition.recognize(itr, paths) if self.recognition else None
        paths = self.reward_handler.merge(paths, critic_rewards, recognition_rewards)
        return self.sampler.process_samples(itr, paths)

    def _save(self, itr):
        """
        Save a tf checkpoint of the session.
        """
        # using keep_checkpoint_every_n_hours as proxy for iterations between saves
        if self.saver and (itr + 1) % self.saver._keep_checkpoint_every_n_hours == 0:
            self.saver.save(
                tf.get_default_session(), 
                self.saver_filepath, 
                global_step=itr
            )

            # save critic and policy params
            save_dir = os.path.join(os.path.split(self.saver_filepath)[0], 'critic')
            critic_params = self.critic.network.get_param_values()
            hgail.misc.utils.save_params(save_dir, critic_params, itr, max_to_keep=50)
            save_dir = os.path.join(os.path.split(self.saver_filepath)[0], 'policy')
            policy_params = self.policy.get_param_values()
            hgail.misc.utils.save_params(save_dir, policy_params, itr, max_to_keep=50)

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
        """
        Snapshot critic and recognition model as well
        """
        self._save(itr)
        self._validate(itr, samples_data)
        snapshot = dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env
        )
        if samples_data is not None:
            snapshot['samples_data'] = dict()
            if 'actions' in samples_data.keys():
                snapshot['samples_data']['actions'] = samples_data['actions'][:10]
            if 'mean' in samples_data.keys():
                snapshot['samples_data']['mean'] = samples_data['mean'][:10]

        return snapshot