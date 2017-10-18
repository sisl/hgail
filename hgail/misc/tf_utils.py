
import tensorflow as tf

def clip_gradients(gradients, rescale, clip):
    gradients, _ = tf.clip_by_global_norm(gradients, rescale)
    gradients = [tf.clip_by_value(g, -clip, clip) for g in gradients]
    return gradients
