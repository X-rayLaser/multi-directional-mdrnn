import numpy as np
import tensorflow as tf

from mdrnn._util.directions import Direction
from mdrnn._util.grids import TensorGrid, PositionOutOfBoundsError


class MDRNN(tf.keras.layers.Layer):
    MAX_INPUT_DIM = 10**10
    MAX_UNITS = MAX_INPUT_DIM
    MAX_NDIMS = 10**3

    def __init__(self, units, input_shape, kernel_initializer=None,
                 recurrent_initializer=None, bias_initializer=None, activation='tanh',
                 return_sequences=False, return_state=False, direction=None, **kwargs):
        if input_shape:
            kwargs.update(dict(input_shape=input_shape))
        super(MDRNN, self).__init__(**kwargs)

        input_dim = input_shape[-1]
        ndims = len(input_shape[:-1])

        if (input_dim <= 0 or input_dim >= self.MAX_INPUT_DIM
                or units <= 0 or units >= self.MAX_UNITS
                or ndims <= 0 or ndims >= self.MAX_NDIMS):
            raise InvalidParamsError()

        if direction is None:
            args = [1] * ndims
            direction = Direction(*args)

        self._validate_direction(direction, ndims)

        self._input_shape = input_shape
        self.ndims = ndims
        self.input_dim = input_dim
        self.units = units
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.direction = direction
        self.activation = tf.keras.activations.get(activation)
        self._kernel_initializer = kernel_initializer
        self._recurrent_initializer = recurrent_initializer
        self._bias_initializer = bias_initializer

        default_initializer = tf.keras.initializers.he_normal()

        if kernel_initializer is None:
            kernel_initializer = default_initializer

        if recurrent_initializer is None:
            recurrent_initializer = default_initializer

        if bias_initializer is None:
            bias_initializer = default_initializer

        self.wax = tf.Variable(kernel_initializer((input_dim, units), dtype=tf.float32))
        self.recurrent_kernels = []

        for _ in range(self.ndims):
            self.recurrent_kernels.append(
                tf.Variable(recurrent_initializer((units, units), dtype=tf.float32))
            )

        self.ba = tf.Variable(bias_initializer((1, units), dtype=tf.float32))

    def spawn(self, direction):
        return MDRNN(units=self.units, input_shape=self._input_shape,
                     kernel_initializer=self._kernel_initializer,
                     recurrent_initializer=self._recurrent_initializer,
                     bias_initializer=self._bias_initializer,
                     activation=self.activation,
                     return_sequences=self.return_sequences,
                     return_state=self.return_state,
                     direction=direction)

    def _validate_direction(self, direction, ndims):
        if not isinstance(direction, Direction):
            raise InvalidParamsError()

        if direction.dimensions != ndims:
            raise InvalidParamsError(direction.dimensions)

    def call(self, inp, initial_state=None, **kwargs):
        self._validate_input(inp)

        if not isinstance(inp, tf.Tensor):
            inp = tf.constant(inp, dtype=tf.float32)

        if inp.dtype != tf.float32:
            inp = tf.cast(inp, dtype=tf.float32)

        initial_state = self._prepare_initial_state(initial_state)
        outputs = self._make_graph(inp, initial_state)

        return self._prepare_result(outputs)

    def _validate_input(self, inp):
        expected_rank = 1 + self.ndims + 1
        if (len(inp.shape) != expected_rank or 0 in inp.shape
                or inp.shape[-1] != self.input_dim):
            raise InputRankMismatchError(inp.shape)

    def _prepare_initial_state(self, initial_state):
        if initial_state is None:
            if self.ndims == 1:
                initial_state = np.zeros((1, self.units), dtype=np.float)
            else:
                initial_state = []
                for i in range(self.ndims):
                    initial_state.append(np.zeros((1, self.units), dtype=np.float))

        if self.ndims == 1:
            initial_state = [initial_state]

        a0 = []
        for i in range(self.ndims):
            a0.append(tf.constant(initial_state[i], dtype=tf.float32))

        return a0

    def _make_graph(self, inp, initial_state):
        grid_shape = self._calculate_grid_shape(inp)

        tensor_shape = (inp.shape[0], self.input_dim)

        outputs = TensorGrid(grid_shape=grid_shape, tensor_shape=tensor_shape)

        positions = self.direction.iterate_over_positions(grid_shape)

        first_position = positions[0]

        for position in positions:
            batch = self._get_batch(inp, position)

            if position == first_position:
                axis_to_activation = [v for v in initial_state]
                axes = list(range(self.ndims))
                z_recurrent = self._compute_weighted_sum_of_activations(
                    axis_to_activation, axes
                )
            else:
                z_recurrent = self._compute_recurrent_weighted_sum(outputs, position)

            z = tf.add(z_recurrent, tf.matmul(batch, self.wax))
            z = tf.add(z, self.ba)
            a = self.activation(z)

            outputs.put_item(position, a)

        return outputs.to_tensor()

    def _calculate_grid_shape(self, inp):
        return inp.shape[1:-1]

    def _get_batch(self, tensor, position):
        i = position[0]
        t = tensor[:, i]

        for index in position[1:]:
            t = t[:, index]
        return t

    def _compute_recurrent_weighted_sum(self, outputs, position):
        previous_positions = self.direction.get_previous_step_positions(position)

        axes_with_positions = self._discard_out_of_bound_positions(
            outputs, previous_positions
        )

        previous_activations = {}
        for axis, prev_position in axes_with_positions:
            previous_activations[axis] = outputs.get_item(prev_position)

        valid_axes = [axis for axis, _ in axes_with_positions]

        return self._compute_weighted_sum_of_activations(previous_activations,
                                                         valid_axes)

    def _compute_weighted_sum_of_activations(self, axis_to_activation, axes):
        z_recurrent = tf.zeros((1, self.units), dtype=tf.float32)
        for axis in axes:
            waa = self.recurrent_kernels[axis]
            a = axis_to_activation[axis]
            z_recurrent = tf.add(tf.matmul(a, waa), z_recurrent)
        return z_recurrent

    def _discard_out_of_bound_positions(self, output_grid, positions):
        valid_positions = []

        for axis, prev_position in enumerate(positions):
            try:
                output_grid.get_item(prev_position)
                valid_positions.append((axis, prev_position))
            except PositionOutOfBoundsError:
                pass

        return valid_positions

    def _prepare_result(self, outputs):
        final_position = self.direction.get_final_position()
        last_state = self._get_batch(outputs, final_position)

        if self.return_sequences:
            returned_outputs = outputs
        else:
            returned_outputs = last_state

        if self.return_state:
            return [returned_outputs] + [last_state]
        return returned_outputs


class InvalidParamsError(Exception):
    pass


class InputRankMismatchError(Exception):
    pass