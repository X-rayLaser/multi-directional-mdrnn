from unittest.case import TestCase

import numpy as np
import tensorflow as tf
from tensorflow_core.python.keras.api._v2.keras import initializers

from mdrnn import MDRNN, MDGRU
from mdrnn._layers.gru import LinearGate


class Degenerate2DInputToMDRNNTests(TestCase):
    def make_rnns(self):
        seed = 1

        kwargs = dict(units=5, input_shape=(None, None, 2),
                      kernel_initializer=initializers.glorot_uniform(seed),
                      recurrent_initializer=initializers.he_normal(seed),
                      bias_initializer=initializers.Constant(2),
                      return_sequences=True,
                      activation='tanh'
                      )

        rnn2d = self.create_mdrnn(**kwargs)

        keras_kwargs = dict(kwargs)
        keras_kwargs.update(dict(input_shape=(None, 2)))
        keras_rnn = self.create_keras_rnn(**keras_kwargs)
        return rnn2d, keras_rnn

    def create_mdrnn(self, **kwargs):
        return MDRNN(**kwargs)

    def create_keras_rnn(self, **kwargs):
        return tf.keras.layers.SimpleRNN(**kwargs)

    def setUp(self):
        self.rnn2d, self.keras_rnn = self.make_rnns()

    def assert_rnn_outputs_equal(self, x_1d, x_2d):
        actual = self.rnn2d(x_2d).numpy()

        desired_shape = x_2d.shape[:-1] + (5, )
        desired = self.keras_rnn(x_1d).numpy().reshape(desired_shape)
        np.testing.assert_almost_equal(actual, desired, 6)

    def test_on_1x1_input(self):
        x = np.array([5, 10], dtype=np.float)
        x_1d = x.reshape((1, 1, 2))
        x_2d = x.reshape((1, 1, 1, 2))

        self.assert_rnn_outputs_equal(x_1d, x_2d)

    def test_on_1xn_input(self):
        x_2d = np.random.rand(3, 1, 4, 2)
        x_1d = x_2d.reshape((3, 4, 2))
        self.assert_rnn_outputs_equal(x_1d, x_2d)

    def test_on_nx1_input(self):
        x_2d = np.random.rand(3, 4, 1, 2)
        x_1d = x_2d.reshape((3, 4, 2))
        self.assert_rnn_outputs_equal(x_1d, x_2d)


class Degenerate2DInputToMDGRUTests(Degenerate2DInputToMDRNNTests):
    def create_mdrnn(self, **kwargs):
        return MDGRU(**kwargs)

    def create_keras_rnn(self, **kwargs):
        return tf.keras.layers.GRU(implementation=1, reset_after=False, **kwargs)


class OutputShapeGiven2DTests(TestCase):
    def get_rnn_class(self):
        return MDRNN

    def test(self):
        RnnClass = self.get_rnn_class()
        rnn2d = RnnClass(units=5, input_shape=(None, None, 1),
                         kernel_initializer=initializers.Constant(1),
                         recurrent_initializer=initializers.Constant(1),
                         bias_initializer=initializers.Constant(-1),
                         return_sequences=True,
                         activation='tanh')

        x = np.arange(6).reshape((1, 2, 3, 1))

        res = rnn2d.call(x)

        self.assertEqual((1, 2, 3, 5), res.shape)


class MDGRUOutputShapeGiven2DTests(OutputShapeGiven2DTests):
    def get_rnn_class(self):
        return MDGRU


class OutputShapeGiven6DInputTests(TestCase):
    def get_rnn_class(self):
        return MDRNN

    def test_shape_of_output_after_processing_6d_input(self):
        units = 7
        last_dim_size = 12
        RnnClass = self.get_rnn_class()
        rnn = RnnClass(units=units, input_shape=(None, None, None, None, None, None, last_dim_size),
                       return_sequences=True,
                       activation='tanh')

        x = tf.zeros(shape=(2, 3, 1, 2, 2, 1, 5, last_dim_size))

        result = rnn.call(x)
        self.assertEqual((2, 3, 1, 2, 2, 1, 5, units), result.shape)


class MDGRUOutputShapeGiven6DInputTests(OutputShapeGiven6DInputTests):
    def get_rnn_class(self):
        return MDGRU


class ProcessingGridInputTests(TestCase):
    def setUp(self):
        RnnClass = self.get_mdrnn_class()
        self.rnn2d = RnnClass(units=1, input_shape=(None, None, 1),
                              kernel_initializer=initializers.Identity(),
                              recurrent_initializer=initializers.Identity(1),
                              bias_initializer=initializers.Constant(-1),
                              return_sequences=True,
                              activation=None)

        self.x = np.arange(6).reshape((1, 2, 3, 1))
        self.initial_state = [tf.ones(shape=(1, 1)), tf.ones(shape=(1, 1))]

    def get_mdrnn_class(self):
        return MDRNN

    def get_expected_result(self):
        return np.array([
            [-1, -1, 0],
            [1, 3, 7]
        ]).reshape((1, 2, 3, 1))

    def get_expected_result_for_processing_with_initial_state(self):
        return np.array([
            [1, 1, 2],
            [3, 7, 13]
        ]).reshape((1, 2, 3, 1))

    def test_result(self):
        actual = self.rnn2d.call(self.x)
        desired = self.get_expected_result()
        np.testing.assert_almost_equal(desired, actual.numpy(), 6)

    def test_2drnn_output_when_providing_initial_state(self):
        actual = self.rnn2d.call(self.x, initial_state=self.initial_state)
        desired = self.get_expected_result_for_processing_with_initial_state()
        np.testing.assert_almost_equal(desired, actual.numpy(), 6)


class ThreeDimensionalInputTests(TestCase):
    def test_result_after_running_rnn_on_3d_input(self):
        rnn3d = MDRNN(units=1, input_shape=(None, None, None, 1),
                      kernel_initializer=initializers.Identity(),
                      recurrent_initializer=initializers.Identity(1),
                      bias_initializer=initializers.Constant(1),
                      return_sequences=True,
                      return_state=True,
                      activation=None)

        x = np.arange(2*2*2).reshape((1, 2, 2, 2, 1))

        outputs, state = rnn3d.call(x)

        desired = np.array([
            [[1, 3],
             [4, 11]],
            [[6, 15],
             [17, 51]]
        ]).reshape((1, 2, 2, 2, 1))

        np.testing.assert_almost_equal(desired, outputs.numpy(), 6)

        desired_state = desired[:, -1, -1, -1]
        np.testing.assert_almost_equal(desired_state, state.numpy(), 6)


class LinearGateExceptionTests(TestCase):
    def setUp(self):
        self.initializer = tf.keras.initializers.zeros()
        self.x = tf.constant(np.random.rand(2, 4), dtype=tf.float64)

    def create_gate(self, **kwargs):
        return LinearGate(kernel_initializer=self.initializer,
                          recurrent_initializer=self.initializer,
                          bias_initializer=self.initializer,
                          **kwargs)

    def test_cannot_use_negative_number_of_dimensions(self):
        self.assertRaises(Exception, lambda: self.create_gate(num_dimensions=-1))

    def test_cannot_use_too_large_number_of_dimensions(self):
        self.assertRaises(Exception,
                          lambda: self.create_gate(num_dimensions=10**3 + 1))

    def test_prev_outputs_cannot_have_more_elements_than_number_of_dimensions(self):
        gate = self.create_gate(units=8, input_size=4, num_dimensions=3)
        self.assertRaises(Exception,
                          lambda: gate.process(self.x, prev_outputs=[3, 2, 1, 3], axes=[0, 1, 2, 3]))

    def test_number_of_axes_cannot_have_more_elements_than_number_of_dimensions(self):
        gate = self.create_gate(units=8, input_size=4, num_dimensions=3)
        self.assertRaises(Exception,
                          lambda: gate.process(self.x, prev_outputs=[3, 2, 1], axes=[0, 1, 2, 3]))


class LinearGateTests(TestCase):
    def setUp(self):
        self.kernel_initializer = tf.keras.initializers.he_normal(seed=1)
        self.recurrent_initializer = tf.keras.initializers.he_normal(seed=2)
        self.bias_initializer = tf.keras.initializers.he_normal(seed=3)

        self.units = 8
        self.input_size = 2

        self.x = tf.constant(np.random.rand(2, self.input_size), dtype=tf.float64)
        self.kernel = self.kernel_initializer((self.input_size, self.units), dtype=tf.float64)
        self.bias = self.bias_initializer((self.units,), dtype=tf.float64)

    def create_gate(self, num_dimensions):
        return LinearGate(units=self.units,
                          input_size=self.input_size,
                          num_dimensions=num_dimensions,
                          kernel_initializer=self.kernel_initializer,
                          recurrent_initializer=self.recurrent_initializer,
                          bias_initializer=self.bias_initializer)

    def test_with_zero_dimensions(self):
        gate = self.create_gate(num_dimensions=0)
        expected = tf.matmul(self.x, self.kernel) + self.bias
        actual = gate.process(self.x, prev_outputs=[], axes=[])
        np.testing.assert_almost_equal(actual.numpy(), expected.numpy(), 6)

    def test_with_one_dimension(self):
        gate = self.create_gate(num_dimensions=1)

        u = self.recurrent_initializer((self.units, self.units), dtype=tf.float64)
        a = tf.constant(np.random.rand(2, self.units), dtype=tf.float64)

        expected = tf.matmul(self.x, self.kernel) + tf.matmul(a, u) + self.bias
        actual = gate.process(self.x, prev_outputs=[a], axes=[0])

        np.testing.assert_almost_equal(actual.numpy(), expected.numpy(), 6)

    def test_with_two_dimensions(self):
        gate = self.create_gate(num_dimensions=2)

        u = self.recurrent_initializer((self.units, self.units * 2), dtype=tf.float64)

        u1 = u[:, :self.units]
        u2 = u[:, self.units:]

        a1 = tf.constant(np.random.rand(2, self.units), dtype=tf.float64)
        a2 = tf.constant(np.random.rand(2, self.units), dtype=tf.float64)

        expected = tf.matmul(self.x, self.kernel) + tf.matmul(a1, u1) + tf.matmul(a2, u2) + self.bias
        actual = gate.process(self.x, prev_outputs=[a1, a2], axes=[0, 1])

        np.testing.assert_almost_equal(actual.numpy(), expected.numpy(), 6)

    def test_correct_recurrent_kernel_is_used(self):
        gate = self.create_gate(num_dimensions=2)

        u = self.recurrent_initializer((self.units, self.units * 2), dtype=tf.float64)

        u1 = u[:, :self.units]
        a1 = tf.constant(np.random.rand(2, self.units), dtype=tf.float64)

        expected = tf.matmul(self.x, self.kernel) + tf.matmul(a1, u1) + self.bias
        actual = gate.process(self.x, prev_outputs={0: a1}, axes=[0])
        np.testing.assert_almost_equal(actual.numpy(), expected.numpy(), 6)

        u2 = u[:, self.units:]
        a2 = tf.constant(np.random.rand(2, self.units), dtype=tf.float64)

        expected = tf.matmul(self.x, self.kernel) + tf.matmul(a2, u2) + self.bias
        actual = gate.process(self.x, prev_outputs={1: a2}, axes=[1])
        np.testing.assert_almost_equal(actual.numpy(), expected.numpy(), 6)


# todo: Finish LinearGate implementation
# todo: Write MDGRUCell in terms of LinearGate objects and test it
# todo: Write shape tests for MultiDirectional(MDGRU(...))
# todo: acceptance/integration tests using MDGRU on dummy data
# todo: update Readme and make a new release
