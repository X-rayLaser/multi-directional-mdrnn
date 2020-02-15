from unittest.case import TestCase

import numpy as np

from mdrnn import MDRNN


class MDRNNOutputShapeTests(TestCase):
    def setUp(self):
        self.x = np.zeros((1, 4, 1))
        self.expected_sequence_shape = (1, 4, 3)
        self.expected_state_shape = (1, 3)

    def create_default_mdrnn(self, **kwargs):
        return MDRNN(units=3, input_shape=(None, 1), **kwargs)

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

    def test_feeding_5_dimensional_rnn_returns_sequences_and_last_state(self):
        rnn = MDRNN(units=3, input_shape=(None, None, None, None, None, 1),
                    return_sequences=True, return_state=True)

        shape = (2, 2, 3, 1, 2, 6, 1)
        x = np.arange(2*2*3*1*2*6).reshape(shape)
        res = rnn(x)
        self.assertEqual(2, len(res))
        sequences, state = res

        expected_sequence_shape = (2, 2, 3, 1, 2, 6, 3)
        expected_state_shape = (2, 3)
        self.assertEqual(expected_sequence_shape, sequences.shape)
        self.assertEqual(expected_state_shape, state.shape)
