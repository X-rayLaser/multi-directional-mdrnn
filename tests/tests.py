from unittest import TestCase
from mdrnn import MDRNN, InvalidParamsError, InputRankMismatchError
import numpy as np
from tensorflow.keras import initializers


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

        self.assert_not_raises(InputRankMismatchError,
                               lambda: mdrnn.call(np.zeros((1, 3, 7))))

        inputs = [np.zeros((2,)), np.zeros((2, 1)), np.zeros((1, 3, 2))]
        self.assert_invalid_input_ranks(mdrnn, *inputs)

    def test_input_array_last_dimension_should_match_input_dim(self):
        mdrnn = MDRNN(input_dim=3, units=1, ndims=1)

        a = np.zeros((1, 2, 3))
        self.assert_not_raises(InputRankMismatchError, lambda: mdrnn.call(a))

        inputs = [np.zeros((1, 2, 2, 1))]
        self.assert_invalid_input_ranks(mdrnn, *inputs)

    def test_input_shape_should_contain_no_zeros(self):
        mdrnn = MDRNN(input_dim=2, units=1, ndims=2)

        inputs = [np.zeros((0, 2, 3)), np.zeros((2, 0, 3)), np.zeros((0, 0, 3))]
        self.assert_invalid_input_ranks(mdrnn, *inputs)


class MDRNNOutputShapeTests(TestCase):
    def setUp(self):
        self.x = np.zeros((1, 4, 1))
        self.expected_sequence_shape = (1, 4, 3)
        self.expected_state_shape = (1, 3)

    def test_feeding_1_dimensional_rnn_returns_sequences(self):
        mdrnn = MDRNN(input_dim=1, units=3, ndims=1, return_sequences=True)
        output = mdrnn.call(self.x)
        self.assertEqual(self.expected_sequence_shape, output.shape)

    def test_feeding_1_dimensional_rnn_returns_last_state(self):
        mdrnn = MDRNN(input_dim=1, units=3, ndims=1)
        self.assertEqual(self.expected_state_shape, mdrnn.call(self.x).shape)

        mdrnn = MDRNN(input_dim=1, units=3, ndims=1, return_sequences=False)
        self.assertEqual(self.expected_state_shape, mdrnn.call(self.x).shape)

    def test_feeding_1_dimensional_rnn_returns_sequences_and_last_state(self):
        mdrnn = MDRNN(input_dim=1, units=3, ndims=1, return_sequences=True, return_state=True)
        res = mdrnn.call(self.x)
        self.assertEqual(2, len(res))
        sequences, state = res

        self.assertEqual(self.expected_sequence_shape, sequences.shape)
        self.assertEqual(self.expected_state_shape, state.shape)

    def test_feeding_1_dimensional_rnn_returns_last_state_in_2_identical_tensors(self):
        mdrnn = MDRNN(input_dim=1, units=3, ndims=1, return_sequences=False, return_state=True)
        res = mdrnn.call(self.x)
        self.assertEqual(2, len(res))
        state, same_state = res
        self.assertEqual(self.expected_state_shape, state.shape)
        self.assertTrue(np.all(state == same_state))

    def test_feeding_batch(self):
        mdrnn = MDRNN(input_dim=1, units=3, ndims=1, return_sequences=True)
        x = np.zeros((2, 4, 1))
        a = mdrnn.call(x)

        self.assertEqual((2, 4, 3), a.shape)


class OneStepTests(TestCase):
    def test_kernel_weights_are_working(self):
        mdrnn = MDRNN(input_dim=3, units=3, ndims=1,
                      kernel_initializer=initializers.Identity(),
                      recurrent_initializer=initializers.Zeros(),
                      bias_initializer=initializers.Zeros(),
                      activation=None,
                      return_sequences=True)

        x1 = np.array([1, 2, 3])
        x = np.array([[x1]])

        a = mdrnn.call(x)

        expected_result = x.copy()
        np.testing.assert_almost_equal(expected_result, a.numpy(), 4)

    def test_recurrent_weights_are_working(self):
        mdrnn = MDRNN(input_dim=3, units=3, ndims=1,
                      kernel_initializer=initializers.Zeros(),
                      recurrent_initializer=initializers.Identity(),
                      bias_initializer=initializers.Zeros(),
                      activation=None)

        initial_state = np.array([1, 2, 3]).reshape(-1, 1)

        import tensorflow as tf
        a = mdrnn.call(np.zeros((1, 1, 3), dtype=np.float),
                       initial_state=tf.constant(initial_state, dtype=tf.float32))

        expected_result = np.array([[1, 2, 3]])
        np.testing.assert_almost_equal(expected_result, a.numpy(), 4)

    def test_biases_are_working(self):
        mdrnn = MDRNN(input_dim=3, units=3, ndims=1,
                      kernel_initializer=initializers.Zeros(),
                      recurrent_initializer=initializers.Zeros(),
                      bias_initializer=initializers.Constant(2),
                      activation=None)

        x = np.zeros((1, 1, 3))
        a = mdrnn.call(x)

        expected_result = np.array([[2, 2, 2]])
        np.testing.assert_almost_equal(expected_result, a.numpy(), 4)

    def test_1d_rnn_produces_correct_output_for_the_first_step(self):
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
        np.testing.assert_almost_equal(expected_result, a.numpy(), 4)

    def test_1d_rnn_with_tensor(self):
        mdrnn = MDRNN(input_dim=3, units=3, ndims=1,
                      kernel_initializer=initializers.Zeros(),
                      recurrent_initializer=initializers.Zeros(),
                      bias_initializer=initializers.Constant(2),
                      activation=None)

        import tensorflow as tf
        x = tf.zeros((1, 1, 3))
        #x = np.zeros((1, 1, 3))
        a = mdrnn.call(x)

        expected_result = np.array([[2, 2, 2]])
        np.testing.assert_almost_equal(expected_result, a.numpy(), 4)


class OneDimensionalRNNTests(TestCase):
    def test_feeding_layer_created_with_default_initializer(self):
        mdrnn = MDRNN(input_dim=1, units=2, ndims=1)
        x = np.zeros((1, 1, 1)) * 0.5
        mdrnn.call(x)

    def test_for_one_step_sequence(self):
        kernel_initializer = initializers.Zeros()
        recurrent_initializer = kernel_initializer

        mdrnn = MDRNN(input_dim=1, units=2, ndims=1,
                      kernel_initializer=kernel_initializer,
                      recurrent_initializer=recurrent_initializer,
                      bias_initializer=initializers.Constant(5),
                      activation=None,
                      return_sequences=True)
        x = np.zeros((1, 1, 1))
        a = mdrnn.call(x)

        expected_result = np.array([
            [[5, 5]]
        ])
        np.testing.assert_almost_equal(expected_result, a.numpy(), 4)

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



# todo 1 dimensional rnn: feed with batch result tests, tensors as inputs in tests, with turned off eager execution
# todo 1 dimensional rnn, specify iteration direction
# todo bidirectional 1 dimensional rnn
# todo 2 dimensional rnn
# todo 2 dimensional rnn 4 directional rnn
# todo n dimensional rnn
# todo n dimensional rnn multidirectional
