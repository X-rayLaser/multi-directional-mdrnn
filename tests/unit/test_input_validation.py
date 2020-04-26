from unittest.case import TestCase

import numpy as np

from mdrnn import MDRNN, MDLSTM, InputRankMismatchError


class MDRNNInputValidationTests(TestCase):
    def make_rnn(self, **kwargs):
        return MDRNN(**kwargs)

    def assert_invalid_input_ranks(self, mdrnn, *numpy_arrays):
        for a in numpy_arrays:
            self.assertRaises(InputRankMismatchError, lambda: mdrnn.call(a))

    def assert_not_raises(self, exc, fn):
        try:
            fn()
        except exc as e:
            self.assertFalse(True, msg='Function raised exception {}'.format(repr(e)))

    def test_input_array_rank_should_be_2_units_larger_than_ndims(self):
        mdrnn = self.make_rnn(units=1, input_shape=(None, 7))

        inputs = [np.zeros((2,)), np.zeros((2, 1)), np.zeros((1, 3, 2))]
        self.assert_invalid_input_ranks(mdrnn, *inputs)

    def test_input_array_last_dimension_should_match_input_dim(self):
        mdrnn = self.make_rnn(units=1, input_shape=(None, 3))

        inputs = [np.zeros((1, 2, 1))]
        self.assert_invalid_input_ranks(mdrnn, *inputs)

    def test_input_shape_should_contain_no_zeros(self):
        mdrnn = self.make_rnn(units=1, input_shape=(None, 1, 2))

        inputs = [np.zeros((0, 2, 3)), np.zeros((2, 0, 3)), np.zeros((0, 0, 3))]
        self.assert_invalid_input_ranks(mdrnn, *inputs)

    def test_with_correct_input(self):
        mdrnn = self.make_rnn(units=1, input_shape=(None, 7))

        self.assert_not_raises(InputRankMismatchError,
                               lambda: mdrnn.call(np.zeros((5, 3, 7))))


class MDGRUInputValidationTests(MDRNNInputValidationTests):
    def make_rnn(self, **kwargs):
        from mdrnn._layers.gru import MDGRU
        return MDGRU(**kwargs)


class MDLSTMInputValidationTests(MDRNNInputValidationTests):
    def make_rnn(self, **kwargs):
        return MDLSTM(**kwargs)
