
import numpy as np
import tensorflow as tf

import rllab.misc.tensor_utils as tensor_utils

import hgail.misc.tf_utils
import hgail.misc.utils

class RecognitionModel(object):

    def __init__(
            self,
            network,
            dataset,
            variable_type,
            latent_dim,
            obs_dim,
            act_dim,
            name='recognition',
            optimizer=tf.train.AdamOptimizer(.0001, beta1=.5, beta2=.9),
            n_train_epochs=5,
            summary_writer=None,
            verbose=0):
        self.network = network
        self.dataset = dataset
        self.variable_type = variable_type
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.name = name
        self.optimizer = optimizer
        self.n_train_epochs = n_train_epochs
        self.summary_writer = summary_writer
        self.verbose = verbose
        if self.variable_type not in ['categorical', 'gaussian']:
            raise NotImplementedError('invalid latent variable type: {}'.format(self.variable_type))
        self._build_model()

    def _extract_actions_latents(self, paths, depth):
        # relevant actions and latent variables may be nested
        acts, latents = [], []
        for d in paths:
            act = d['actions']
            latent_info = d['agent_infos']['latent_info']
            
            for _ in range(depth):
                act = latent_info['latent']
                latent_info = latent_info['latent_info']

            acts.append(act)
            latents.append(latent_info['latent'])
        return acts, latents

    def _normalize(self, obs, acts):
        if self.dataset.observation_normalizer:
            obs = self.dataset.observation_normalizer(acts)
        if self.dataset.action_normalizer:
            acts = self.dataset.action_normalizer(acts)
        return obs, acts

    def _extract_from_paths(self, paths, depth):
        # relevant actions and latent variables may be nested
        acts, latents = self._extract_actions_latents(paths, depth)

        # convert to batch and run recognition
        obs = np.concatenate([d['observations'] for d in paths], axis=0)
        acts = np.concatenate(acts, axis=0)
        latents = np.concatenate(latents, axis=0)

        # optionally normalize
        obs, acts = self._normalize(obs, acts)

        return obs, acts, latents

    def recognize(self, itr, paths, depth=0):
        """
        Compute and return rewards based on the (obs, action) pairs in paths
            where rewards are a list of numpy arrays of equal length as the 
            corresponding path rewards
        
        Args:
            itr: iteration count
            paths: list of dictionaries
        """
        obs, acts, latents = self._extract_from_paths(paths, depth)
        
        if self.variable_type == 'categorical':
            probs = self._probs(obs, acts)
            idxs = np.argmax(latents, axis=1)
            rewards = probs[np.arange(len(idxs)), idxs]
        elif self.variable_type == 'gaussian':
            probs = self._probs(obs, acts, latents)
            rewards = probs

        # output as a list of numpy arrays, each of len equal to the rewards of 
        # the corresponding trajectory
        path_lengths = [len(d['rewards']) for d in paths]
        path_rewards = hgail.misc.utils.batch_to_path_rewards(rewards, path_lengths)

        self._log_recognize(itr, paths, rewards)
        return path_rewards

    def _probs(self, obs, act, latents=None):
        """
        Compute the probability of the latent variable values given 
        the observations and actions

        Args:
            - obs: numpy array shape (batch_size, obs_dim)
            - act: numpy array shape (batch_size, act_dim)
        """
        feed_dict = {self.x: obs, self.a: act}
        if latents is not None:
            feed_dict[self.c] = latents
        session = tf.get_default_session()
        probs = session.run(self.probs, feed_dict=feed_dict)
        return probs

    def _log_recognize(self, itr, paths, recognition_rewards):
        """
        Log information about the critique and paths

        Args:
            itr: algorithm batch iteration
            paths: list of dictionaries containing trajectory information
            critic_rewards: critic rewards
        """
        # only write summaries if have a summary writer
        if self.summary_writer:

            summary = tf.Summary(value=[
                tf.Summary.Value(
                    tag="recognition/mean_recognition_reward", 
                    simple_value=np.mean(recognition_rewards)), 
                tf.Summary.Value(
                    tag="recognition/std_dev_recognition_reward", 
                    simple_value=np.std(recognition_rewards))
            ])
            self.summary_writer.add_summary(summary, itr)
            self.summary_writer.flush()

    def train(self, itr, samples_data):
        """
        Train the recognition model to identify the latent variable
        underlying the (state, action) pair
        
        Args:
            itr: iteration count
            samples_data: dictionary containing generated data
        """
        for train_itr in range(self.n_train_epochs):
            for batch in self.dataset.batches(samples_data):
                self._train_batch(batch)

    def _train_batch(self, batch):
        """
        Runs a single training batch
        
        Args:
            batch: dictionary with values needed for training network class member
        """
        feed_dict = {
            self.x: batch['x'],
            self.a: batch['a'],
            self.c: batch['c']
        }
        outputs_list = [self.train_op, self.summary_op, self.global_step]
        session = tf.get_default_session()
        _, summary, step = session.run(outputs_list, feed_dict=feed_dict)

        if self.summary_writer:
            self.summary_writer.add_summary(tf.Summary.FromString(summary), step)
            self.summary_writer.flush()

    def _build_model(self):
        """
        Builds the recognition model loss and train operation
        """
        self._build_placeholders()
        self._forward()
        self._build_loss()
        self._build_train_op()
        self._build_summaries()

    def _build_placeholders(self):
        self.x = tf.placeholder(tf.float32, shape=(None, self.obs_dim), name='x')
        self.a = tf.placeholder(tf.float32, shape=(None, self.act_dim), name='a')
        self.c = tf.placeholder(tf.float32, shape=(None, self.latent_dim), name='c')
        self.global_step = tf.Variable(0, name='recognition_model/global_step', trainable=False)

    def _forward(self):
        self.scores = self.network(self.x, self.a)

        if self.variable_type == 'categorical':
            self.probs = tf.nn.softmax(self.scores)

        elif self.variable_type == 'gaussian':
            # in the gaussian case, the recognition network needs to output 
            # both the means and std deviations for the gaussian distribution
            if self.scores.shape[1] != 2 * self.latent_dim:
                raise ValueError(
                    'when using gaussian latent variables '
                    'the recognition network must be created with 2x '
                    'the output size, but it is {} when it should be {}'.format(
                        self.scores.shape[1], 2 * self.latent_dim
                    )
                )
            self.mean, self.logvar = tf.split(self.scores, 2, axis=1)
            self.sigma = tf.exp(self.logvar / 2.)
            self.dist = tf.contrib.distributions.MultivariateNormalDiag(self.mean, self.sigma)
            self.probs = self.dist.prob(self.c)
            self.log_probs = self.dist.log_prob(self.c)

    def _build_loss(self):
        if self.variable_type == 'categorical':
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.c, logits=self.scores)
            self.loss = tf.reduce_mean(loss)
        elif self.variable_type == 'gaussian':
            loss = -self.log_probs
            self.loss = tf.reduce_mean(loss)
        if self.verbose >= 2:
            self.loss = tf.Print(self.loss, [self.loss], message='recognition loss: ')

    def _build_train_op(self):
        self.gradients = gradients = tf.gradients(self.loss, self.network.var_list)
        grads_vars = [(g,v) for (g,v) in zip(gradients, self.network.var_list)]
        self.train_op = self.optimizer.apply_gradients(grads_vars, global_step=self.global_step)

    def _build_summaries(self):
        summaries = []
        summaries += [tf.summary.scalar('{}/loss'.format(self.name), self.loss)]
        summaries += [tf.summary.scalar('{}/mean_c'.format(self.name), 
            tf.reduce_mean(self.c))]
        summaries += [tf.summary.scalar('{}/mean_probs'.format(self.name), 
            tf.reduce_mean(self.probs))]
        summaries += [tf.summary.scalar('{}/global_grad_norm'.format(self.name), 
            tf.global_norm(self.gradients))]
        summaries += [tf.summary.scalar('{}/global_var_norm'.format(self.name), 
            tf.global_norm(self.network.var_list))]
        self.summaries = summaries
        self.summary_op = tf.summary.merge(self.summaries)
        
        