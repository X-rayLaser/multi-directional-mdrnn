from unittest.case import TestCase

import numpy as np

from mdrnn import MDRNN, MDLSTM, MultiDirectional


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


class MDLSTMOutputShapeTests(TestCase):
    def setUp(self):
        self.x = np.zeros((2, 4, 1))
        self.expected_sequence_shape = (2, 4, 3)
        self.expected_state_shape = (2, 3)

    def create_default_mdrnn(self, **kwargs):
        return MDLSTM(units=3, input_shape=(None, 1), **kwargs)

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
        self.assertEqual(3, len(res))
        sequences, a, c = res

        self.assertEqual(self.expected_sequence_shape, sequences.shape)
        self.assertEqual(self.expected_state_shape, a.shape)
        self.assertEqual(self.expected_state_shape, c.shape)

    def test_feeding_1_dimensional_rnn_returns_last_state_in_2_identical_tensors(self):
        mdrnn = self.create_default_mdrnn(return_sequences=False, return_state=True)
        res = mdrnn.call(self.x)
        self.assertEqual(3, len(res))
        outputs, a, c = res
        self.assertEqual(self.expected_state_shape, outputs.shape)
        self.assertTrue(np.all(outputs == a))

    def test_feeding_5_dimensional_rnn_returns_sequences_and_last_state(self):
        rnn = MDLSTM(units=3, input_shape=(None, None, None, None, None, 1),
                     return_sequences=True, return_state=True)

        shape = (2, 2, 3, 1, 2, 6, 1)
        x = np.arange(2*2*3*1*2*6).reshape(shape)
        res = rnn(x)
        self.assertEqual(3, len(res))
        sequences, a, c = res

        expected_sequence_shape = (2, 2, 3, 1, 2, 6, 3)
        expected_state_shape = (2, 3)
        self.assertEqual(expected_sequence_shape, sequences.shape)
        self.assertEqual(expected_state_shape, a.shape)
        self.assertEqual(expected_state_shape, c.shape)


class MultiDirectional2DLSTMOutputShapeTests(TestCase):
    def setUp(self):
        self.units = 6
        self.x = np.zeros((2, 3, 4, 5))
        self.expected_sequence_shape = (2, 3, 4, self.units * 4)
        self.last_activation_shape = (2, self.units * 4)
        self.expected_state_shape = (2, self.units)
        self.num_state_vectors = 8

    def create_default_mdrnn(self, **kwargs):
        return MultiDirectional(MDLSTM(units=6, input_shape=(None, None, 5), **kwargs))

    def test_feeding_2_dimensional_rnn_returns_sequences(self):
        mdrnn = self.create_default_mdrnn(return_sequences=True)
        output = mdrnn.call(self.x)
        self.assertEqual(self.expected_sequence_shape, output.shape)

    def test_feeding_2_dimensional_rnn_returns_last_state(self):
        mdrnn = self.create_default_mdrnn()
        self.assertEqual(self.last_activation_shape, mdrnn.call(self.x).shape)

        mdrnn = self.create_default_mdrnn(return_sequences=False)
        self.assertEqual(self.last_activation_shape, mdrnn.call(self.x).shape)

    def test_feeding_2_dimensional_rnn_returns_sequences_and_last_state(self):
        mdrnn = self.create_default_mdrnn(return_sequences=True, return_state=True)
        res = mdrnn.call(self.x)
        self.assertEqual(self.num_state_vectors + 1, len(res))
        sequences = res[0]
        states = res[1:]

        self.assertEqual(self.expected_sequence_shape, sequences.shape)

        for s in states:
            self.assertEqual(self.expected_state_shape, s.shape)
        self.assertEqual(len(states), self.num_state_vectors)

    def test_output_tensor_equals_concatenation_of_state_activations(self):
        mdrnn = self.create_default_mdrnn(return_sequences=False, return_state=True)
        res = mdrnn(self.x)
        self.assertEqual(self.num_state_vectors + 1, len(res))
        outputs = res[0]
        states = res[1:]

        np_vectors = [s.numpy() for s in states[::2]]

        self.assertEqual(len(np_vectors), self.num_state_vectors // 2)

        self.assertTrue(np.all(outputs == np.hstack(np_vectors)))

        for s in np_vectors:
            self.assertEqual(s.shape, self.expected_state_shape)


# todo: remove duplication among test case classes
