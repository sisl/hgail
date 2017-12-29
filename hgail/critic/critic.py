
import numpy as np
import tensorflow as tf

import hgail.misc.utils
import hgail.misc.tf_utils

class Critic(object):
    """
    Critic base class
    """
    def __init__(
            self,
            network,
            dataset,
            obs_dim, 
            act_dim,
            optimizer=tf.train.RMSPropOptimizer(0.0001),
            n_train_epochs=5,
            grad_norm_rescale=10000.,
            grad_norm_clip=10000.,
            summary_writer=None,
            debug_nan=False,
            verbose=0):
        self.network = network
        self.dataset = dataset
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.optimizer = optimizer
        self.n_train_epochs = n_train_epochs
        self.grad_norm_rescale = grad_norm_rescale
        self.grad_norm_clip = grad_norm_clip
        self.summary_writer = summary_writer
        self.debug_nan = debug_nan
        self.verbose = verbose
    
    def critique(self, itr, paths):
        """
        Compute and return rewards based on the (obs, action) pairs in paths
            where rewards are a list of numpy arrays of equal length as the 
            corresponding path rewards
        
        Args:
            itr: iteration count
            paths: list of dictionaries
        """
        # convert to batch and use network to critique
        obs = np.concatenate([d['observations'] for d in paths], axis=0)
        acts = np.concatenate([d['actions'] for d in paths], axis=0)

        # normalize
        if self.dataset.observation_normalizer:
            obs = self.dataset.observation_normalizer(obs)
        if self.dataset.action_normalizer:
            acts = self.dataset.action_normalizer(acts)

        # compute rewards
        rewards = self.network.forward(obs, acts, deterministic=True)

        if np.any(np.isnan(rewards)) and self.debug_nan:
            import ipdb
            ipdb.set_trace()
        
        # output as a list of numpy arrays, each of len equal to the rewards of 
        # the corresponding trajectory
        path_lengths = [len(d['rewards']) for d in paths]
        path_rewards = hgail.misc.utils.batch_to_path_rewards(rewards, path_lengths)

        self._log_critique(itr, paths, rewards)
        return path_rewards

    def _log_critique(self, itr, paths, critic_rewards):
        """
        Log information about the critique and paths

        Args:
            itr: algorithm batch iteration
            paths: list of dictionaries containing trajectory information
            critic_rewards: critic rewards
        """
        # only write summaries if have a summary writer
        if self.summary_writer:

            env_rewards = np.concatenate([d['rewards'] for d in paths], axis=0)
            summary = tf.Summary(value=[
                tf.Summary.Value(tag="critic/mean_critique_reward", simple_value=np.mean(critic_rewards)),
                tf.Summary.Value(tag="critic/max_critique_reward", simple_value=np.max(critic_rewards)), 
                tf.Summary.Value(tag="critic/std_dev_critique_reward", simple_value=np.std(critic_rewards)), 
                tf.Summary.Value(tag="critic/mean_env_reward", simple_value=np.mean(env_rewards)),
                tf.Summary.Value(tag="critic/mean_path_len", simple_value=len(env_rewards) / float(len(paths))),
            ])
            self.summary_writer.add_summary(summary, itr)
            self.summary_writer.flush()

    def train(self, itr, samples_data):
        """
        Train the critic using real and sampled data
        
        Args:
            itr: iteration count
            samples_data: dictionary containing generated data
        """
        for train_itr in range(self.n_train_epochs):
            for batch in self.dataset.batches(samples_data, store=train_itr == 0):
                self._train_batch(batch)
            
    def _train_batch(self, batch):
        """
        Runs a single training batch
        
        Args:
            batch: dictionary with values needed for training network class member
        """
        raise NotImplementedError()

    def _build_summaries(
            self, 
            loss, 
            real_loss,
            gen_loss, 
            gradients, 
            clipped_gradients, 
            gradient_penalty=None,
            batch_size=None):
        summaries = []
        summaries += [tf.summary.scalar('critic/loss', loss)]
        summaries += [tf.summary.scalar('critic/w_dist', -(real_loss + gen_loss))]
        summaries += [tf.summary.scalar('critic/real_loss', real_loss)]
        summaries += [tf.summary.scalar('critic/gen_loss', gen_loss)]
        summaries += [tf.summary.scalar('critic/global_grad_norm', tf.global_norm(gradients))]
        summaries += [tf.summary.scalar('critic/global_clipped_grad_norm', tf.global_norm(clipped_gradients))]
        summaries += [tf.summary.scalar('critic/global_var_norm', tf.global_norm(self.network.var_list))]
        if gradient_penalty is not None:
            summaries += [tf.summary.scalar('critic/gradient_penalty', gradient_penalty)]
        if batch_size is not None:
            summaries += [tf.summary.scalar('critic/batch_size', batch_size)]
        return summaries

    def _build_input_summaries(self, rx, ra, gx, ga):
        summaries = []
        summaries += [tf.summary.image('critic/real_obs', tf.reshape(rx[0], (-1, self.obs_dim, 1, 1)))]
        summaries += [tf.summary.image('critic/real_act', tf.reshape(ra[0], (-1, self.act_dim, 1, 1)))]
        summaries += [tf.summary.image('critic/gen_obs', tf.reshape(gx[0], (-1, self.obs_dim, 1, 1)))]
        summaries += [tf.summary.image('critic/gen_act', tf.reshape(ga[0], (-1, self.act_dim, 1, 1)))]
        return summaries

class WassersteinCritic(Critic):
    
    def __init__(
            self,
            gradient_penalty=10.,
            **kwargs):
        super(WassersteinCritic, self).__init__(**kwargs)
        self.gradient_penalty = gradient_penalty
        self._build_placeholders()
        self._build_model()

    def _build_placeholders(self):
        # placeholders for input
        self.rx = tf.placeholder(tf.float32, shape=(None, self.obs_dim), name='rx')
        self.ra = tf.placeholder(tf.float32, shape=(None, self.act_dim), name='ra')
        self.gx = tf.placeholder(tf.float32, shape=(None, self.obs_dim), name='gx')
        self.ga = tf.placeholder(tf.float32, shape=(None, self.act_dim), name='ga')
        self.eps  = tf.placeholder(tf.float32, shape=(None, 1), name='eps')

    def _build_model(self):
        # unpack placeholders
        rx, ra, gx, ga, eps = self.rx, self.ra, self.gx, self.ga, self.eps

        # gradient penalty        
        self.xhat = xhat = eps * rx + (1 - eps) * gx
        self.ahat = ahat = eps * ra + (1 - eps) * ga
        xhat_gradients, ahat_gradients = tf.gradients(self.network(xhat, ahat), [xhat, ahat])
        self.hat_gradients = hat_gradients = tf.concat([xhat_gradients, ahat_gradients], axis=1)
        slopes = tf.sqrt(tf.reduce_sum(hat_gradients ** 2, reduction_indices=[1]) + 1e-8)
        self.gp_loss = gp_loss = self.gradient_penalty * tf.reduce_mean((slopes - 1) ** 2)
        
        # loss and train op
        self.real_loss = real_loss = -tf.reduce_mean(self.network(rx, ra))
        self.gen_loss = gen_loss = tf.reduce_mean(self.network(gx, ga))
        self.loss = loss = real_loss + gen_loss + gp_loss

        if self.verbose >= 2:
            loss = tf.Print(loss, [real_loss, gen_loss, gp_loss, loss],
                message='real, gen, gp, total loss: ')
        
        self.gradients = gradients = tf.gradients(loss, self.network.var_list)
        clipped_gradients = hgail.misc.tf_utils.clip_gradients(
            gradients, self.grad_norm_rescale, self.grad_norm_clip)
        
        self.global_step = tf.Variable(0, name='critic/global_step', trainable=False)
        self.train_op = self.optimizer.apply_gradients([(g,v) 
                            for (g,v) in zip(clipped_gradients, self.network.var_list)],
                            global_step=self.global_step)
        
        # summaries
        summaries = self._build_summaries(loss, real_loss, gen_loss, gradients, clipped_gradients, gp_loss)
        summaries += self._build_input_summaries(rx, ra, gx, ga)
        self.summary_op = tf.summary.merge(summaries)

        # debug_nan
        self.gp_gradients = tf.gradients(self.gp_loss, self.network.var_list)[:-1]
        
    def _train_batch(self, batch):

        feed_dict = {
            self.rx: batch['rx'],
            self.ra: batch['ra'],
            self.gx: batch['gx'],
            self.ga: batch['ga'],
            self.eps: np.random.uniform(0, 1, len(batch['rx'])).reshape(-1, 1)
        }
        outputs_list = [self.train_op, self.summary_op, self.global_step]
        if self.debug_nan:
            outputs_list += [
                self.gradients, 
                self.xhat, 
                self.ahat, 
                self.hat_gradients,
                self.gp_gradients,
                self.gp_loss,
                self.real_loss, 
                self.gen_loss 
            ] 
        session = tf.get_default_session()
        fetched = session.run(outputs_list, feed_dict=feed_dict)
        summary, step = fetched[1], fetched[2]

        if self.debug_nan:
            grads, xhat, ahat, hat_grads, gp_grads, gp_loss, real_loss, gen_loss = fetched[3:]
            grads_nan = np.any([np.any(np.isnan(g)) for g in grads])

            if grads_nan or np.isnan(gp_loss) or np.isnan(real_loss) or np.isnan(gen_loss):
                import ipdb
                ipdb.set_trace()

        if self.summary_writer:
            self.summary_writer.add_summary(tf.Summary.FromString(summary), step)
            self.summary_writer.flush()
