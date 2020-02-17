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

    def test_with_functor(self):
        rnn = self.create_mdrnn(direction=Direction(1))

        rnn = MultiDirectional(rnn)

        a = rnn(self.x, initial_state=self.initial_state)

        expected_result = np.array([
            [2, 8],
            [4, 4],
            [8, 2]
        ]).reshape((1, 3, 2))

        np.testing.assert_almost_equal(expected_result, a.numpy(), 8)


class UniDirectionalRnnTests(TestCase):
    def setUp(self):
        self.x = tf.constant(np.random.rand(3, 4, 5), dtype=tf.float32)

    def make_rnns(self, return_sequences, return_state):
        seed = 1
        kwargs = dict(units=3, input_shape=(None, 5),
                      kernel_initializer=initializers.glorot_uniform(seed),
                      recurrent_initializer=initializers.he_normal(seed),
                      bias_initializer=initializers.Constant(2),
                      return_sequences=return_sequences,
                      return_state=return_state,
                      activation='relu'
                      )
        rnn = MDRNN(**kwargs)

        keras_rnn = tf.keras.layers.SimpleRNN(**kwargs)
        return rnn, keras_rnn

    def test_when_return_sequences_and_return_state_are_false(self):
        rnn, keras_rnn = self.make_rnns(return_sequences=False, return_state=False)
        np.testing.assert_almost_equal(rnn(self.x).numpy(), keras_rnn(self.x).numpy(), 6)

    def test_when_return_sequences_is_true(self):
        rnn, keras_rnn = self.make_rnns(return_sequences=True, return_state=False)

        np.testing.assert_almost_equal(rnn(self.x).numpy(), keras_rnn(self.x).numpy(), 6)

    def test_when_return_state_is_true(self):
        rnn, keras_rnn = self.make_rnns(return_sequences=False, return_state=True)
        self.assert_outputs_of_neural_nets_equal(rnn, keras_rnn)

    def test_when_both_return_sequences_and_return_state_are_true(self):
        rnn, keras_rnn = self.make_rnns(return_sequences=True, return_state=True)
        self.assert_outputs_of_neural_nets_equal(rnn, keras_rnn)

    def assert_outputs_of_neural_nets_equal(self, rnn, keras_rnn):
        rnn_result = rnn(self.x)
        rnn_output = rnn_result[0]
        rnn_state = rnn_result[1]

        keras_rnn_result = keras_rnn(self.x)
        keras_rnn_output = keras_rnn_result[0]
        keras_rnn_state = keras_rnn_result[1]

        self.assertEqual(len(rnn_result), len(keras_rnn_result))

        np.testing.assert_almost_equal(rnn_output.numpy(), keras_rnn_output.numpy(), 6)
        np.testing.assert_almost_equal(rnn_state.numpy(), keras_rnn_state.numpy(), 6)

    def test_when_return_state_is_true_rnn_returns_list_instead_of_tuple(self):
        rnn, _ = self.make_rnns(return_sequences=False, return_state=True)
        res = rnn.call(self.x)
        self.assertIsInstance(res, list)

        rnn, _ = self.make_rnns(return_sequences=True, return_state=True)
        res = rnn.call(self.x)
        self.assertIsInstance(res, list)


class BiDirectionalTests(TestCase):
    def setUp(self):
        self.x = tf.constant(np.random.rand(3, 4, 5), dtype=tf.float32)

    def make_rnns(self, return_sequences, return_state):
        seed = 1
        kwargs = dict(units=3, input_shape=(None, 5),
                      kernel_initializer=initializers.glorot_uniform(seed),
                      recurrent_initializer=initializers.he_normal(seed),
                      bias_initializer=initializers.Constant(2),
                      return_sequences=return_sequences,
                      return_state=return_state,
                      activation='relu'
                      )
        rnn = MultiDirectional(MDRNN(**kwargs))

        keras_rnn = tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(**kwargs))
        return rnn, keras_rnn

    def test_when_return_sequences_and_return_state_are_false(self):
        rnn, keras_rnn = self.make_rnns(return_sequences=False, return_state=False)
        np.testing.assert_almost_equal(rnn(self.x).numpy(), keras_rnn(self.x).numpy(), 6)

    def test_when_return_sequences_is_true(self):
        rnn, keras_rnn = self.make_rnns(return_sequences=True, return_state=False)

        np.testing.assert_almost_equal(rnn(self.x).numpy(), keras_rnn(self.x).numpy(), 6)

    def assert_outputs_of_neural_nets_equal(self, rnn, keras_rnn):
        rnn_result = rnn(self.x)
        rnn_output = rnn_result[0]
        rnn_state1 = rnn_result[1]
        rnn_state2 = rnn_result[2]

        keras_rnn_result = keras_rnn(self.x)
        keras_rnn_output = keras_rnn_result[0]
        keras_rnn_state1 = keras_rnn_result[1]
        keras_rnn_state2 = keras_rnn_result[2]

        np.testing.assert_almost_equal(rnn_output.numpy(), keras_rnn_output.numpy(), 6)
        np.testing.assert_almost_equal(rnn_state1.numpy(), keras_rnn_state1.numpy(), 6)
        np.testing.assert_almost_equal(rnn_state2.numpy(), keras_rnn_state2.numpy(), 6)

    def test_when_return_state_is_true(self):
        rnn, keras_rnn = self.make_rnns(return_sequences=False, return_state=True)
        self.assert_outputs_of_neural_nets_equal(rnn, keras_rnn)

    def test_when_both_return_sequences_and_return_state_are_true(self):
        rnn, keras_rnn = self.make_rnns(return_sequences=True, return_state=True)
        self.assert_outputs_of_neural_nets_equal(rnn, keras_rnn)
