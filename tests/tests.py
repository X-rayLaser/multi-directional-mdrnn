from unittest import TestCase
from mdrnn import MDRNN, InvalidParamsError, InputRankMismatchError
import numpy as np
from tensorflow.keras import initializers


class MDRNNTests(TestCase):
    def assert_invalid_instances(self, *kwargs):
        for kwargs in kwargs:
            self.assertRaises(InvalidParamsError, lambda: MDRNN(**kwargs))

    def assert_invalid_input_ranks(self, mdrnn, *numpy_arrays):
        for a in numpy_arrays:
            self.assertRaises(InputRankMismatchError, lambda: mdrnn(a))

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

    def test_input_array_rank_should_be_one_unit_larger_than_ndims(self):
        mdrnn = MDRNN(input_dim=7, units=1, ndims=1)

        inputs = [np.zeros((2,)), np.zeros((3, 3, 2))]
        self.assert_invalid_input_ranks(mdrnn, *inputs)

    def test_input_array_last_dimension_should_match_input_dim(self):
        mdrnn = MDRNN(input_dim=3, units=1, ndims=2)

        inputs = [np.zeros((1, 2, 1))]
        self.assert_invalid_input_ranks(mdrnn, *inputs)

    def test_input_shape_should_contain_no_zeros(self):
        mdrnn = MDRNN(input_dim=3, units=1, ndims=2)

        inputs = [np.zeros((0, 2, 3)), np.zeros((2, 0, 3)), np.zeros((0, 0, 3))]
        self.assert_invalid_input_ranks(mdrnn, *inputs)

    def test_feeding_layer_created_with_default_initializer(self):
        mdrnn = MDRNN(input_dim=1, units=2, ndims=1)
        x = np.zeros((1, 1)) * 0.5
        a = mdrnn(x)
        np.testing.assert_almost_equal(np.zeros((1, 2)), a.numpy(), 4)

    def test_for_one_step_sequence(self):
        kernel_initializer = initializers.Zeros()
        recurrent_initializer = kernel_initializer

        mdrnn = MDRNN(input_dim=1, units=2, ndims=1,
                      kernel_initializer=kernel_initializer,
                      recurrent_initializer=recurrent_initializer,
                      bias_initializer=initializers.Constant(5))
        x = np.zeros((1, 1))
        a = mdrnn(x)

        np.testing.assert_almost_equal(np.array([[0.9999, 0.9999]]), a.numpy(), 4)

    def test_for_two_step_sequence(self):
        kernel_initializer = initializers.Zeros()
        recurrent_initializer = kernel_initializer

        mdrnn = MDRNN(input_dim=1, units=2, ndims=1,
                      kernel_initializer=kernel_initializer,
                      recurrent_initializer=recurrent_initializer,
                      bias_initializer=initializers.Constant(5))
        x = np.zeros((2, 1))
        a = mdrnn(x)

        np.testing.assert_almost_equal(np.array([[0.9999, 0.9999], [0.9999, 0.9999]]), a.numpy(), 4)

    def test_with_identity_matrices(self):
        kernel_initializer = initializers.identity()
        recurrent_initializer = kernel_initializer
        bias_initializer = initializers.Zeros()

        mdrnn = MDRNN(input_dim=3, units=3, ndims=1,
                      kernel_initializer=kernel_initializer,
                      recurrent_initializer=recurrent_initializer,
                      bias_initializer=bias_initializer,
                      activation=None)

        x1 = np.array([1, 2, 4])
        x2 = np.array([9, 8, 6])
        x = np.array([x1, x2])
        a = mdrnn(x)

        expected_result = np.array([x1, x1 + x2])
        np.testing.assert_almost_equal(expected_result, a.numpy(), 4)
