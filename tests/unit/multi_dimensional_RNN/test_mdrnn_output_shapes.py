from unittest import TestCase
import numpy as np
from mdrnn import MDRNN, MultiDirectional


class ShapeOfMultiDirectional2DRnnTests(TestCase):
    def setUp(self):
        self.x = np.zeros((3, 4, 2, 5))
        self.units = 6

    @property
    def batch_size(self):
        return self.x.shape[0]

    @property
    def grid_shape(self):
        return self.x.shape[1:-1]

    @property
    def num_dimensions(self):
        return len(self.grid_shape)

    @property
    def num_directions(self):
        return 2 ** self.num_dimensions

    def make_rnn(self, return_sequences, return_state):
        shape = self.x.shape[1:]
        rnn = MDRNN(units=self.units, input_shape=shape,
                    return_sequences=return_sequences,
                    return_state=return_state,
                    activation='tanh')

        return MultiDirectional(rnn)

    def assert_state_shapes_equal(self, expected_shape, states):
        for i in range(len(states)):
            self.assertEqual(expected_shape, states[i].shape)

    def test_when_return_sequences_and_return_states_is_false(self):
        rnn = self.make_rnn(return_sequences=False, return_state=False)
        res = rnn(self.x)

        self.assertEqual((self.batch_size, self.units * self.num_directions), res.shape)

    def test_when_return_sequences_is_false_but_return_states_is_true(self):
        rnn = self.make_rnn(return_sequences=False, return_state=True)
        res = rnn(self.x)
        output = res[0]
        states = res[1:]
        self.assertEqual((self.batch_size, self.units * self.num_directions), output.shape)
        self.assert_state_shapes_equal((self.batch_size, self.units), states)

    def test_when_return_sequences_is_true_but_return_states_is_false(self):
        rnn = self.make_rnn(return_sequences=True, return_state=False)
        output = rnn(self.x)

        expected_shape = (self.batch_size,) + \
                         self.grid_shape + \
                         (self.units * self.num_directions,)

        self.assertEqual(expected_shape, output.shape)

    def test_when_both_return_sequences_and_return_states_is_true(self):
        rnn = self.make_rnn(return_sequences=True, return_state=True)
        res = rnn(self.x)
        output = res[0]
        states = res[1:]

        expected_output_shape = (self.batch_size,) + \
                                self.grid_shape + \
                                (self.units * self.num_directions,)

        self.assertEqual(expected_output_shape, output.shape)
        self.assert_state_shapes_equal((self.batch_size, self.units), states)


class ShapeOfMultiDirectional3DRnnTests(ShapeOfMultiDirectional2DRnnTests):
    def setUp(self):
        self.x = np.zeros((3, 4, 2, 2, 5))
        self.units = 6
