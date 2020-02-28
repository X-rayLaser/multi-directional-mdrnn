from unittest import TestCase
from tensorflow.keras import initializers
import tensorflow as tf
import numpy as np


class UnidirectionalOneDimensionalGRUTests(TestCase):
    def make_rnn(self, return_sequences, return_state):
        seed = 1
        kwargs = dict(units=3, input_shape=(None, 5),
                      kernel_initializer=initializers.glorot_uniform(seed),
                      recurrent_initializer=initializers.he_normal(seed),
                      bias_initializer=initializers.he_normal(seed),
                      return_sequences=return_sequences,
                      return_state=return_state
                      )
        rnn = MDGRU(**kwargs)
        keras_rnn = tf.keras.layers.GRU(reset_after=True, implementation=1, **kwargs)
        return rnn, keras_rnn
