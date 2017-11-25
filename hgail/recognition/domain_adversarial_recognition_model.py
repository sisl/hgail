'''
pass d in env info
pass this to the dataset
dataset passes it in in the batch
ok so then I have d now what
now add adv loss
how?
need features from the network
so create a network that returns the features
done
'''
import tensorflow as tf

from hgail.recognition.recognition_model import RecognitionModel
from hgail.misc.flip_gradient import flip_gradient

class DomainAdvRecognitionModel(RecognitionModel):

    def __init__(self,
                latent_classifier,
                domain_classifier,
                domain_dim=2,
                lambda_initial=0.,
                lambda_steps=1000,
                lambda_final=1.,
                grad_clip=1000.,
                grad_scale=1000.,
                *args,
                **kwargs):
        self.latent_classifier = latent_classifier
        self.domain_classifier = domain_classifier
        self.domain_dim = domain_dim
        self.lambda_initial = lambda_initial
        self.lambda_steps = lambda_steps
        self.lambda_final = lambda_final
        self.grad_clip = grad_clip
        self.grad_scale = grad_scale
        super(DomainAdvRecognitionModel, self).__init__(latent_classifier, *args, **kwargs)

    def _train_batch(self, batch):
        """
        Runs a single training batch
        
        Args:
            batch: dictionary with values needed for training network class member
        """
        feed_dict = {
            self.x: batch['x'],
            self.a: batch['a'],
            self.c: batch['c'],
            self.d: batch['d']
        }
        outputs_list = [self.train_op, self.summary_op, self.global_step]
        session = tf.get_default_session()
        _, summary, step = session.run(outputs_list, feed_dict=feed_dict)

        if self.summary_writer:
            self.summary_writer.add_summary(tf.Summary.FromString(summary), step)
            self.summary_writer.flush()

    def _build_placeholders(self):
        super(DomainAdvRecognitionModel, self)._build_placeholders()
        self.d = tf.placeholder(tf.float32, shape=(None, self.domain_dim), name='d')
        self.lmbda = tf.train.polynomial_decay(
            self.lambda_initial, 
            self.global_step, 
            self.lambda_steps, 
            end_learning_rate=self.lambda_final, 
            power=2.0,
            name='lambda'
        )

    def _forward(self):
        self.scores, all_features = self.latent_classifier(self.x, self.a)
        self.features = all_features[-1]
        self.probs = tf.nn.softmax(self.scores)
        flipped_features = flip_gradient(self.features, self.lmbda)
        self.domain_scores = self.domain_classifier(flipped_features)
        # self.domain_scores = self.domain_classifier(tf.stop_gradient(self.features))
        self.domain_probs = tf.nn.softmax(self.domain_scores)

    def _build_loss(self):
        super(DomainAdvRecognitionModel, self)._build_loss()
        self.pred_loss = self.loss
        self.reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        self.domain_adv_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    labels=self.d, logits=self.domain_scores))
        self.loss = self.pred_loss + self.reg_loss + self.domain_adv_loss
        
        # domain accuracy
        pred = tf.cast(tf.argmax(self.domain_scores, axis=-1), tf.float32)
        true = tf.cast(tf.argmax(self.d, axis=-1), tf.float32)
        self.domain_acc = tf.reduce_mean(tf.cast(tf.equal(pred, true), tf.float32))

        # latent accuracy
        pred = tf.cast(tf.argmax(self.scores, axis=-1), tf.float32)
        true = tf.cast(tf.argmax(self.c, axis=-1), tf.float32)
        self.acc = tf.reduce_mean(tf.cast(tf.equal(pred, true), tf.float32))

        if self.verbose >= 2:
            self.loss = tf.Print(self.loss, 
                [self.loss, self.acc, self.domain_acc], 
                message='data_loss, domain_loss, acc, domain acc: ')

    def _build_train_op(self):
        self.all_network_var_list = self.latent_classifier.var_list + self.domain_classifier.var_list
        self.gradients = tf.gradients(self.loss, self.all_network_var_list)
        self.clipped_gradients, _ = tf.clip_by_global_norm(self.gradients, self.grad_scale)
        clip_grads_vars = [(tf.clip_by_value(g, -self.grad_clip, self.grad_clip),v)
            for (g,v) in zip(self.clipped_gradients, self.all_network_var_list)]
        self.clipped_gradients = [g for (g,_) in clip_grads_vars]
        self.train_op = self.optimizer.apply_gradients(
            clip_grads_vars, global_step=self.global_step)

    def _build_summaries(self):
        super(DomainAdvRecognitionModel, self)._build_summaries()
        self.summaries += [tf.summary.scalar('{}/pred_loss'.format(self.name), self.pred_loss)]
        self.summaries += [tf.summary.scalar('{}/domain_adv_loss'.format(self.name), self.domain_adv_loss)]
        self.summaries += [tf.summary.scalar('{}/lambda'.format(self.name), self.lmbda)]
        self.summaries += [tf.summary.scalar('{}/domain_acc'.format(self.name), self.domain_acc)]
        self.summaries += [tf.summary.scalar('{}/acc'.format(self.name), self.acc)]
        self.summaries += [tf.summary.scalar('{}/clip_grad_norm'.format(self.name), 
            tf.global_norm(self.clipped_gradients))]
        self.summary_op = tf.summary.merge(self.summaries)
