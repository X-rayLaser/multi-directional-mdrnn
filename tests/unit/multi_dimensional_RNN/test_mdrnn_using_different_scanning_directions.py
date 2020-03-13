from unittest import TestCase
import numpy as np
from mdrnn import MDRNN, Direction, MultiDirectional
from tensorflow.keras import initializers


class RnnMovingSouthEast(TestCase):
    def setUp(self):
        self.rnn_setup = Rnn2dTestSetup(direction=self.get_direction())
        self.rnn = self.rnn_setup.make_rnn()
        self.x = self.rnn_setup.make_input()

    def get_direction(self):
        return Direction(1, 1)

    def assert_rnn_outputs_are_correct(self, output, state):
        expected_result = self.rnn_setup.get_expected_result()
        np.testing.assert_almost_equal(expected_result, output.numpy(), 6)

        expected_state = self.rnn_setup.get_expected_last_state()
        np.testing.assert_almost_equal(expected_state, state.numpy(), 6)

    def test_2d_rnn_scanning_in_given_direction(self):
        output, state = self.rnn.call(self.x)
        self.assert_rnn_outputs_are_correct(output, state)

    def test_2d_rnn_scanning_in_given_direction_using_functor(self):
        output, state = self.rnn(self.x)
        self.assert_rnn_outputs_are_correct(output, state)


class RnnMovingSouthWest(RnnMovingSouthEast):
    def get_direction(self):
        return Direction(1, -1)


class RnnMovingNorthEast(RnnMovingSouthEast):
    def get_direction(self):
        return Direction(-1, 1)


class RnnMovingNorthWest(RnnMovingSouthEast):
    def get_direction(self):
        return Direction(-1, -1)



class OutputOfMultiDirectional2DrnnTests(TestCase):
    def test_returns_list_of_correct_length(self):
        rnn_setup = Rnn2dTestSetup(direction=Direction.south_east())
        rnn = rnn_setup.make_rnn()

        rnn = MultiDirectional(rnn)
        x = rnn_setup.make_input()
        actual = rnn.call(x)

        # 1 element for output of RNN and 4 elements for states, 1 state per each direction
        self.assertEqual(5, len(actual))

    def test_results(self):
        rnn_setup = Rnn2dTestSetup(direction=Direction.south_east())
        rnn = rnn_setup.make_rnn()

        rnn = MultiDirectional(rnn)
        x = rnn_setup.make_input()
        expected = rnn_setup.get_expected_result_for_multi_directional_rnn()
        actual = rnn.call(x)

        num_elements = 5
        for i in range(num_elements):
            actual_output = actual[i]
            expected_output = expected[i]
            np.testing.assert_almost_equal(expected_output, actual_output.numpy(), 6)



class Rnn2dTestSetup:
    south_east_output = np.array([
        [-1, -1, 0],
        [1, 3, 7]
    ]).reshape((1, 2, 3, 1))

    south_west_output = np.array([
        [0, 1, 1],
        [11, 9, 5]
    ]).reshape((1, 2, 3, 1))

    north_east_output = np.array([
        [1, 6, 16],
        [2, 5, 9]
    ]).reshape((1, 2, 3, 1))

    north_west_output = np.array([
        [20, 12, 5],
        [9, 7, 4]
    ]).reshape((1, 2, 3, 1))

    south_east_state = np.array([7]).reshape((1, 1))
    south_west_state = np.array([11]).reshape((1, 1))
    north_east_state = np.array([16]).reshape((1, 1))
    north_west_state = np.array([20]).reshape((1, 1))

    def __init__(self, direction):
        self._direction = direction

    def make_rnn(self):
        kwargs = dict(units=1, input_shape=(None, None, 1),
                      kernel_initializer=initializers.Identity(),
                      recurrent_initializer=initializers.Identity(1),
                      bias_initializer=initializers.Constant(-1),
                      return_sequences=True,
                      return_state=True,
                      direction=self._direction,
                      activation=None)

        return MDRNN(**kwargs)

    def make_input(self):
        return np.arange(6).reshape((1, 2, 3, 1))

    def get_expected_result(self):
        if self._direction == Direction.south_east():
            return self.south_east_output
        elif self._direction == Direction.south_west():
            return self.south_west_output
        elif self._direction == Direction.north_east():
            return self.north_east_output
        elif self._direction == Direction.north_west():
            return self.north_west_output

    @staticmethod
    def get_expected_result_for_multi_directional_rnn():
        outputs = [
            Rnn2dTestSetup.south_east_output,
            Rnn2dTestSetup.south_west_output,
            Rnn2dTestSetup.north_east_output,
            Rnn2dTestSetup.north_west_output
         ]

        states = [
            Rnn2dTestSetup.south_east_state,
            Rnn2dTestSetup.south_west_state,
            Rnn2dTestSetup.north_east_state,
            Rnn2dTestSetup.north_west_state
        ]
        return [np.concatenate(outputs, axis=3)] + states

    def get_expected_last_state(self):
        if self._direction == Direction.south_east():
            return self.south_east_state
        elif self._direction == Direction.south_west():
            return self.south_west_state
        elif self._direction == Direction.north_east():
            return self.north_east_state
        elif self._direction == Direction.north_west():
            return self.north_west_state
