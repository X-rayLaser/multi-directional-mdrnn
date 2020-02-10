from unittest.case import TestCase

import numpy as np
from tensorflow.keras import initializers

from mdrnn import MDRNN, Direction, MultiDirectional
import tensorflow as tf


class ForwardBackwardProcessingTests(TestCase):
    def setUp(self):
        self.x = np.ones((1, 3, 1))
        self.initial_state = np.ones((1, 1))
        self.forward_result = [2, 4, 8]
        self.backward_result = list(reversed(self.forward_result))

    def create_mdrnn(self, direction):
        kernel_initializer = initializers.zeros()
        recurrent_initializer = initializers.constant(2)
        bias_initializer = initializers.zeros()

        return MDRNN(units=1, input_shape=(None, 1),
                     kernel_initializer=kernel_initializer,
                     recurrent_initializer=recurrent_initializer,
                     bias_initializer=bias_initializer,
                     activation=None,
                     return_sequences=True,
                     direction=direction)

    def test_forward_direction(self):
        mdrnn = self.create_mdrnn(direction=Direction(1))

        a = mdrnn.call(self.x, initial_state=self.initial_state, dtype=np.float)

        expected_result = np.array(self.forward_result).reshape((1, 3, 1))

        np.testing.assert_almost_equal(expected_result, a.numpy(), 8)

    def test_backward_direction(self):
        mdrnn = self.create_mdrnn(direction=Direction(-1))

        a = mdrnn.call(self.x, initial_state=self.initial_state, dtype=np.float)

        expected_result = np.array(self.backward_result).reshape((1, 3, 1))

        np.testing.assert_almost_equal(expected_result, a.numpy(), 8)

    def test_bidirectional_rnn_returns_result_with_correct_shape(self):
        from mdrnn import MultiDirectional

        rnn = self.create_mdrnn(direction=Direction(1))

        rnn = MultiDirectional(rnn)

        a = rnn.call(self.x, initial_state=self.initial_state, dtype=np.float)
        self.assertEqual((1, 3, 2), a.shape)

    def test_bidirectional_rnn_returns_correct_result(self):
        rnn = self.create_mdrnn(direction=Direction(1))

        rnn = MultiDirectional(rnn)

        a = rnn.call(self.x, initial_state=self.initial_state, dtype=np.float)

        expected_result = np.array([
            [2, 8],
            [4, 4],
            [8, 2]
        ]).reshape((1, 3, 2))

        np.testing.assert_almost_equal(expected_result, a.numpy(), 8)

    def test(self):
        seed = 1
        kwargs = dict(units=3, input_shape=(None, 5),
                      kernel_initializer=initializers.glorot_uniform(seed),
                      recurrent_initializer=initializers.he_normal(seed),
                      bias_initializer=initializers.Constant(2),
                      return_sequences=True,
                      activation='relu'
                      )

        rnn = MultiDirectional(MDRNN(**kwargs))

        keras_rnn = tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(**kwargs))

        x = tf.constant(np.random.rand(3, 4, 5), dtype=tf.float32)

        np.testing.assert_almost_equal(rnn(x).numpy(), keras_rnn(x).numpy(), 6)