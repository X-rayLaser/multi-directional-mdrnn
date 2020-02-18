from unittest import TestCase
import numpy as np
from mdrnn import MDRNN, MultiDirectional, Direction

from tensorflow.keras import initializers
import tensorflow as tf


class Degenerate2DInputTests(TestCase):
    def setUp(self):
        seed = 1

        kwargs = dict(units=5, input_shape=(None, None, 2),
                      kernel_initializer=initializers.glorot_uniform(seed),
                      recurrent_initializer=initializers.he_normal(seed),
                      bias_initializer=initializers.Constant(2),
                      return_sequences=True,
                      activation='tanh'
                      )

        self.rnn2d = MDRNN(**kwargs)

        keras_kwargs = dict(kwargs)
        keras_kwargs.update(dict(input_shape=(None, 2)))
        self.keras_rnn = tf.keras.layers.SimpleRNN(**keras_kwargs)

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


class GridInputTests(TestCase):
    def test_shape(self):
        rnn2d = MDRNN(units=5, input_shape=(None, None, 1),
                      kernel_initializer=initializers.Constant(1),
                      recurrent_initializer=initializers.Constant(1),
                      bias_initializer=initializers.Constant(-1),
                      return_sequences=True,
                      activation='tanh')

        x = np.arange(6).reshape((1, 2, 3, 1))

        res = rnn2d.call(x)

        self.assertEqual((1, 2, 3, 5), res.shape)

    def test_result(self):
        rnn2d = MDRNN(units=1, input_shape=(None, None, 1),
                      kernel_initializer=initializers.Identity(),
                      recurrent_initializer=initializers.Identity(1),
                      bias_initializer=initializers.Constant(-1),
                      return_sequences=True,
                      activation=None)

        x = np.arange(6).reshape((1, 2, 3, 1))

        actual = rnn2d.call(x)

        desired = np.array([
            [-1, -1, 0],
            [1, 3, 7]
        ]).reshape((1, 2, 3, 1))

        np.testing.assert_almost_equal(desired, actual.numpy(), 6)

    def test_2drnn_output_when_providing_initial_state(self):
        rnn2d = MDRNN(units=1, input_shape=(None, None, 1),
                      kernel_initializer=initializers.Identity(),
                      recurrent_initializer=initializers.Identity(1),
                      bias_initializer=initializers.Constant(-1),
                      return_sequences=True,
                      activation=None)

        x = np.arange(6).reshape((1, 2, 3, 1))

        initial_state = [tf.ones(shape=(1, 1)), tf.ones(shape=(1, 1))]

        actual = rnn2d.call(x, initial_state=initial_state)
        desired = np.array([
            [1, 1, 2],
            [3, 7, 13]
        ]).reshape((1, 2, 3, 1))
        np.testing.assert_almost_equal(desired, actual.numpy(), 6)

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

    def test_shape_of_output_of_6d_rnn(self):
        units = 7
        last_dim_size = 12
        rnn = MDRNN(units=units, input_shape=(None, None, None, None, None, None, last_dim_size),
                    return_sequences=True,
                    activation='tanh')

        x = tf.zeros(shape=(2, 3, 1, 2, 2, 1, 5, last_dim_size))

        result = rnn.call(x)
        self.assertEqual((2, 3, 1, 2, 2, 1, 5, units), result.shape)


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
