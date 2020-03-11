from unittest import TestCase
from tensorflow.keras import initializers
import tensorflow as tf
import numpy as np
from mdrnn import MDGRU, Direction


class UnidirectionalOneDimensionalGRUTests(TestCase):
    def make_rnn(self, return_sequences, return_state, go_backwards=False):
        seed = 1
        kernel_initializer = initializers.glorot_uniform(seed)
        recurrent_initializer = initializers.he_normal(seed)
        bias_initializer = initializers.he_normal(seed)

        kwargs = dict(units=3, input_shape=(None, 5),
                      activation='tanh',
                      recurrent_activation='sigmoid',
                      kernel_initializer=kernel_initializer,
                      recurrent_initializer=recurrent_initializer,
                      bias_initializer=bias_initializer,
                      return_sequences=return_sequences,
                      return_state=return_state
                      )

        if go_backwards:
            kwargs.update(dict(direction=Direction(-1)))

        rnn = MDGRU(**kwargs)
        keras_rnn = tf.keras.layers.GRU(units=3, activation='tanh',
                                        recurrent_activation='sigmoid', implementation=1,
                                        kernel_initializer=kernel_initializer,
                                        recurrent_initializer=recurrent_initializer,
                                        bias_initializer=bias_initializer,
                                        reset_after=False,
                                        go_backwards=go_backwards)
        return rnn, keras_rnn

    def test_outputs_on_single_step_sequence(self):
        rnn, keras_rnn = self.make_rnn(return_sequences=False, return_state=False)

        self.x = tf.constant(np.random.rand(3, 1, 5), dtype=tf.float32)

        rnn_result = rnn(self.x)

        keras_rnn_result = keras_rnn(self.x)

        self.assertEqual(len(rnn_result), len(keras_rnn_result))

        np.testing.assert_almost_equal(rnn_result.numpy(), keras_rnn_result.numpy(), 6)

    def test_outputs_on_many_steps_sequence(self):
        rnn, keras_rnn = self.make_rnn(return_sequences=False, return_state=False)

        self.x = tf.constant(np.random.rand(3, 4, 5), dtype=tf.float32)

        rnn_result = rnn(self.x)

        keras_rnn_result = keras_rnn(self.x)

        self.assertEqual(len(rnn_result), len(keras_rnn_result))

        np.testing.assert_almost_equal(rnn_result.numpy(), keras_rnn_result.numpy(), 6)

    def test_processing_sequence_backwards(self):
        rnn, keras_rnn = self.make_rnn(return_sequences=False, return_state=False, go_backwards=True)

        self.x = tf.constant(np.random.rand(3, 4, 5), dtype=tf.float32)

        rnn_result = rnn(self.x)

        keras_rnn_result = keras_rnn(self.x)

        self.assertEqual(len(rnn_result), len(keras_rnn_result))

        np.testing.assert_almost_equal(rnn_result.numpy(), keras_rnn_result.numpy(), 6)

