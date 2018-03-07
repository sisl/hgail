
import io
import tensorflow as tf

import sys
import matplotlib
backend = 'Agg' if sys.platform == 'linux' else 'TkAgg'
matplotlib.use(backend)
import matplotlib.pyplot as plt

def plt2imgsum():
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_sum = tf.Summary.Image(encoded_image_string=buf.getvalue())
    plt.clf()
    return img_sum

def simple_value_summary(value, tag):
    return tf.Summary.Value(tag=tag, simple_value=value)

class Validator(object):

    def __init__(self, writer):
        self.writer = writer

    def write_summaries(self, itr, summaries):
        summary = tf.Summary(value=summaries)
        self.writer.add_summary(summary, itr)

    def validate(self, itr, objs):
        raise NotImplementedError("validate must be implemented")