from unittest import TestCase
from mdrnn import MDRNN, InvalidParamsError, InputRankMismatchError, Direction
import numpy as np
from tensorflow.keras import initializers
import tensorflow as tf


class MDRNNInitializationTests(TestCase):
    def assert_invalid_instances(self, *kwargs):
        for kwargs in kwargs:
            self.assertRaises(InvalidParamsError, lambda: MDRNN(**kwargs))

    def test_with_invalid_input_dim(self):
        self.assert_invalid_instances(dict(input_dim=-1, units=10, ndims=2),
                                      dict(input_dim=0, units=10, ndims=2),
                                      dict(input_dim=10**10, units=10, ndims=2))

    def test_with_invalid_units(self):
        self.assert_invalid_instances(dict(input_dim=1, units=-1, ndims=2),
                                      dict(input_dim=1, units=0, ndims=2),
                                      dict(input_dim=1, units=10**10, ndims=2))

    def test_with_invalid_number_of_dimensions(self):
        self.assert_invalid_instances(dict(input_dim=1, units=1, ndims=-1),
                                      dict(input_dim=1, units=1, ndims=0),
                                      dict(input_dim=1, units=1, ndims=10**3))


class MDRNNInputValidationTests(TestCase):
    def assert_invalid_input_ranks(self, mdrnn, *numpy_arrays):
        for a in numpy_arrays:
            self.assertRaises(InputRankMismatchError, lambda: mdrnn.call(a))

    def assert_not_raises(self, exc, fn):
        try:
            fn()
        except exc as e:
            self.assertFalse(True, msg='Function raised exception {}'.format(repr(e)))

    def test_input_array_rank_should_be_2_units_larger_than_ndims(self):
        mdrnn = MDRNN(input_dim=7, units=1, ndims=1)

        inputs = [np.zeros((2,)), np.zeros((2, 1)), np.zeros((1, 3, 2))]
        self.assert_invalid_input_ranks(mdrnn, *inputs)

    def test_input_array_last_dimension_should_match_input_dim(self):
        mdrnn = MDRNN(input_dim=3, units=1, ndims=1)

        inputs = [np.zeros((1, 2, 2, 1))]
        self.assert_invalid_input_ranks(mdrnn, *inputs)

    def test_input_shape_should_contain_no_zeros(self):
        mdrnn = MDRNN(input_dim=2, units=1, ndims=2)

        inputs = [np.zeros((0, 2, 3)), np.zeros((2, 0, 3)), np.zeros((0, 0, 3))]
        self.assert_invalid_input_ranks(mdrnn, *inputs)

    def test_with_correct_input(self):
        mdrnn = MDRNN(input_dim=7, units=1, ndims=1)

        self.assert_not_raises(InputRankMismatchError,
                               lambda: mdrnn.call(np.zeros((5, 3, 7))))


class MDRNNOutputShapeTests(TestCase):
    def setUp(self):
        self.x = np.zeros((1, 4, 1))
        self.expected_sequence_shape = (1, 4, 3)
        self.expected_state_shape = (1, 3)

    def create_default_mdrnn(self, **kwargs):
        return MDRNN(input_dim=1, units=3, ndims=1, **kwargs)

    def test_feeding_1_dimensional_rnn_returns_sequences(self):
        mdrnn = self.create_default_mdrnn(return_sequences=True)
        output = mdrnn.call(self.x)
        self.assertEqual(self.expected_sequence_shape, output.shape)

    def test_feeding_1_dimensional_rnn_returns_last_state(self):
        mdrnn = self.create_default_mdrnn()
        self.assertEqual(self.expected_state_shape, mdrnn.call(self.x).shape)

        mdrnn = self.create_default_mdrnn(return_sequences=False)
        self.assertEqual(self.expected_state_shape, mdrnn.call(self.x).shape)

    def test_feeding_1_dimensional_rnn_returns_sequences_and_last_state(self):
        mdrnn = self.create_default_mdrnn(return_sequences=True, return_state=True)
        res = mdrnn.call(self.x)
        self.assertEqual(2, len(res))
        sequences, state = res

        self.assertEqual(self.expected_sequence_shape, sequences.shape)
        self.assertEqual(self.expected_state_shape, state.shape)

    def test_feeding_1_dimensional_rnn_returns_last_state_in_2_identical_tensors(self):
        mdrnn = self.create_default_mdrnn(return_sequences=False, return_state=True)
        res = mdrnn.call(self.x)
        self.assertEqual(2, len(res))
        state, same_state = res
        self.assertEqual(self.expected_state_shape, state.shape)
        self.assertTrue(np.all(state == same_state))

    def test_feeding_batch(self):
        mdrnn = self.create_default_mdrnn(return_sequences=True)
        x = np.zeros((2, 4, 1))
        a = mdrnn.call(x)

        self.assertEqual((2, 4, 3), a.shape)


class OneStepTests(TestCase):
    def setUp(self):
        self.x = np.zeros((1, 1, 3), dtype=np.float)

    def create_default_mdrnn(self, **kwargs):
        return MDRNN(input_dim=3, units=3, ndims=1, **kwargs)

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


class TwoStepRNNTests(TestCase):
    def test_feeding_layer_created_with_default_initializer(self):
        mdrnn = MDRNN(input_dim=1, units=2, ndims=1)
        x = np.zeros((1, 1, 1)) * 0.5
        mdrnn.call(x)

    def test_for_two_step_sequence(self):
        kernel_initializer = initializers.Zeros()
        recurrent_initializer = kernel_initializer

        mdrnn = MDRNN(input_dim=1, units=2, ndims=1,
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

        mdrnn = MDRNN(input_dim=3, units=3, ndims=1,
                      kernel_initializer=kernel_initializer,
                      recurrent_initializer=recurrent_initializer,
                      bias_initializer=bias_initializer,
                      activation=None,
                      return_sequences=True)

        x1 = np.array([1, 2, 4])
        x2 = np.array([9, 8, 6])
        x = np.array([x1, x2])
        a = mdrnn.call(x.reshape(1, 2, -1))

        expected_result = np.array([[x1 + bias, x1 + x2 + 2 * bias]])
        np.testing.assert_almost_equal(expected_result, a.numpy(), 8)


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

        return MDRNN(input_dim=1, units=1, ndims=1,
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


class DirectionValidationTestCase(TestCase):
    def setUp(self):
        self.input_dim = 3
        self.units = 3
        self.ndims = 1

    def create_mdrnn(self, direction):
        return MDRNN(input_dim=self.input_dim, units=self.units,
                     ndims=self.ndims, direction=direction)

    def assert_invalid_direction(self, direction):
        self.assertRaises(
            InvalidParamsError,
            lambda: self.create_mdrnn(direction=direction)
        )

    def assert_valid_direction(self, direction):
        try:
            self.create_mdrnn(direction=direction)
        except InvalidParamsError:
            self.assertTrue(False, msg='Direction is not valid')


class DirectionValidationTests(DirectionValidationTestCase):
    def test_cannot_use_integer_for_direction(self):
        self.assert_invalid_direction(3)

    def test_cannot_use_list_for_direction(self):
        self.assert_invalid_direction([1, -1, -1])

    def test_length_of_direction_must_match_dimensionality_of_input(self):
        self.assert_invalid_direction(Direction())
        self.assert_invalid_direction(Direction(1, 0))

    def test_with_valid_direction(self):
        self.assert_valid_direction(Direction(-1))


class DirectionValidationInMultidimensionalRNN(DirectionValidationTestCase):
    def setUp(self):
        self.input_dim = 3
        self.units = 3
        self.ndims = 2

    def test_pass_invalid_direction_to_2drnn(self):
        self.assert_invalid_direction(Direction(1))

    def test_with_valid_direction(self):
        self.assert_valid_direction(Direction(-1, 1))


# todo 1 dimensional rnn, specify iteration direction
# todo bidirectional 1 dimensional rnn
# todo 2 dimensional rnn
# todo 2 dimensional rnn 4 directional rnn
# todo n dimensional rnn
# todo n dimensional rnn multidirectional
