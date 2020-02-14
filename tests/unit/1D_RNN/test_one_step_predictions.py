from unittest.case import TestCase

import numpy as np
import tensorflow as tf
from tensorflow_core.python.keras.api._v2.keras import initializers

from mdrnn import MDRNN


class OneStepTests(TestCase):
    def setUp(self):
        self.x = np.zeros((1, 1, 3), dtype=np.float)

    def create_default_mdrnn(self, **kwargs):
        return MDRNN(units=3, input_shape=(None, 3), **kwargs)

    def assert_model_predicts_correct_result(self, expected, mdrnn, inputs, **kwargs):
        a = mdrnn.call(inputs, **kwargs)
        np.testing.assert_almost_equal(expected, a.numpy(), 8)

    def assert_arrays_equal(self, expected, actual):
        np.testing.assert_almost_equal(expected, actual.numpy(), 8)

    def test_kernel_weights_are_working(self):
        mdrnn = self.create_default_mdrnn(
            kernel_initializer=initializers.Identity(),
            recurrent_initializer=initializers.Zeros(),
            bias_initializer=initializers.Zeros(),
            activation=None,
            return_sequences=True
        )

        x1 = np.array([1, 2, 3])
        x = np.array([[x1]])

        a = mdrnn.call(x)

        expected_result = x.copy()
        np.testing.assert_almost_equal(expected_result, a.numpy(), 8)

    def test_recurrent_weights_are_working(self):
        mdrnn = self.create_default_mdrnn(
            kernel_initializer=initializers.Zeros(),
            recurrent_initializer=initializers.Identity(),
            bias_initializer=initializers.Zeros(),
            activation=None
        )

        initial_state = np.array([1, 2, 3]).reshape(1, -1)
        initial_state = tf.constant(initial_state, dtype=tf.float32)

        expected_result = np.array([[1, 2, 3]])
        self.assert_model_predicts_correct_result(expected_result, mdrnn, self.x,
                                                  initial_state=initial_state)

    def test_biases_are_working(self):
        mdrnn = self.create_default_mdrnn(
            kernel_initializer=initializers.Zeros(),
            recurrent_initializer=initializers.Zeros(),
            bias_initializer=initializers.Constant(2),
            activation=None
        )

        expected_result = np.array([[2, 2, 2]])
        self.assert_model_predicts_correct_result(expected_result, mdrnn, self.x)

    def test_activation_is_working(self):
        initializer = initializers.zeros()

        mdrnn = self.create_default_mdrnn(
            kernel_initializer=initializer,
            recurrent_initializer=initializer,
            bias_initializer=initializer,
            activation='sigmoid'
        )

        expected_result = np.ones((1, 3)) * 0.5
        self.assert_model_predicts_correct_result(expected_result, mdrnn, self.x)

    def test_1d_rnn_with_tensor(self):
        mdrnn = self.create_default_mdrnn(
            kernel_initializer=initializers.Zeros(),
            recurrent_initializer=initializers.Zeros(),
            bias_initializer=initializers.Constant(2),
            activation=None
        )

        x = tf.constant(self.x, dtype=tf.float32)
        expected_result = np.array([[2, 2, 2]])
        self.assert_model_predicts_correct_result(expected_result, mdrnn, x)

    def test_1d_rnn_using_functor(self):
        mdrnn = self.create_default_mdrnn(
            kernel_initializer=initializers.Zeros(),
            recurrent_initializer=initializers.Zeros(),
            bias_initializer=initializers.Constant(2),
            activation=None
        )

        x = tf.constant(self.x, dtype=tf.float32)
        expected_result = np.array([[2, 2, 2]])

        a = mdrnn(x)
        np.testing.assert_almost_equal(expected_result, a.numpy(), 8)
