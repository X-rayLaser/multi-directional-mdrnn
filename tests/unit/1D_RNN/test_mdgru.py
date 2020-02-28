from unittest import TestCase
from tensorflow.keras import initializers
from mdrnn import MDGRU
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

    def test(self):
        return
        rnn, keras_rnn = self.make_rnn(return_sequences=False, return_state=False)

        x = np.zeros((1, 1, 5))
        np.testing.assert_almost_equal(keras_rnn(x).numpy(), rnn(x).numpy())
