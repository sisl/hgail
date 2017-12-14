
import tensorflow as tf

class Validator(object):

    def __init__(self, writer):
        self.writer = writer

    def write_summaries(self, itr, summaries):
        summary = tf.Summary(value=summaries)
        self.writer.add_summary(summary, itr)

    def validate(self, itr, objs):
        raise NotImplementedError("validate must be implemented")