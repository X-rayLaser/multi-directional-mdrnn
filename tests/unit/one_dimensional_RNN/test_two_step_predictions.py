from unittest.case import TestCase

import numpy as np
from tensorflow.keras import initializers
import tensorflow as tf
from mdrnn import MDRNN


class TwoStepRNNTests(TestCase):
    def test_feeding_layer_created_with_default_initializer(self):
        mdrnn = MDRNN(units=2, input_shape=(None, 1))
        x = np.zeros((1, 1, 1)) * 0.5
        mdrnn.call(x)

    def test_for_two_step_sequence(self):
        kernel_initializer = initializers.Zeros()
        recurrent_initializer = kernel_initializer

        mdrnn = MDRNN(units=2, input_shape=(None, 1),
                      kernel_initializer=kernel_initializer,
                      recurrent_initializer=recurrent_initializer,
                      bias_initializer=initializers.Constant(5),
                      return_sequences=True)
        x = np.zeros((1, 3, 1))
        a = mdrnn.call(x)

        expected_result = np.ones((1, 3, 2)) * 0.9999
        np.testing.assert_almost_equal(expected_result, a.numpy(), 4)

    def test_1d_rnn_produces_correct_output_for_2_steps(self):
        kernel_initializer = initializers.identity()
        recurrent_initializer = kernel_initializer

        bias = 3
        bias_initializer = initializers.Constant(bias)

        mdrnn = MDRNN(units=3, input_shape=(None, 3),
                      kernel_initializer=kernel_initializer,
                      recurrent_initializer=recurrent_initializer,
                      bias_initializer=bias_initializer,
                      activation=None,
                      return_sequences=True)

        x1 = np.array([1, 2, 4])
        x2 = np.array([9, 8, 6])
        x = np.array([x1, x2])
        a = mdrnn.call(
            x.reshape((1, 2, -1))
        )

        expected_result = np.array([[x1 + bias, x1 + x2 + 2 * bias]])
        np.testing.assert_almost_equal(expected_result, a.numpy(), 8)


class CheckingMDRNNOutputAgainstKerasSimpleRNN(TestCase):
    def setUp(self):
        seed = 1

        kwargs = dict(units=3, input_shape=(None, 5),
                      kernel_initializer=initializers.glorot_uniform(seed),
                      recurrent_initializer=initializers.he_normal(seed),
                      bias_initializer=initializers.Constant(2),
                      return_sequences=True,
                      activation='relu'
                      )
        self.kwargs = kwargs

    def test_output_sequences_match(self):
        self.kwargs.update(dict(return_sequences=True))
        rnn = MDRNN(**self.kwargs)
        keras_rnn = tf.keras.layers.SimpleRNN(**self.kwargs)

        x = tf.constant(np.random.rand(3, 4, 5), dtype=tf.float32)

        np.testing.assert_almost_equal(rnn(x).numpy(), keras_rnn(x).numpy(), 6)

    def test_hidden_states_match(self):
        self.kwargs.update(dict(return_sequences=False))

        rnn = MDRNN(**self.kwargs)
        keras_rnn = tf.keras.layers.SimpleRNN(**self.kwargs)

        x = tf.constant(np.random.rand(3, 4, 5), dtype=tf.float32)

        np.testing.assert_almost_equal(rnn(x).numpy(), keras_rnn(x).numpy(), 6)
