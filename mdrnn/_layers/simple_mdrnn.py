import numpy as np
import tensorflow as tf

from mdrnn._util.directions import Direction
from mdrnn._util.grids import TensorGrid, PositionOutOfBoundsError


class RnnValidator:
    MAX_INPUT_DIM = 10**10
    MAX_UNITS = MAX_INPUT_DIM
    MAX_NDIMS = 10**3

    @staticmethod
    def validate_rnn_params(units, input_dim, ndims):
        if (input_dim <= 0 or input_dim >= RnnValidator.MAX_INPUT_DIM
                or units <= 0 or units >= RnnValidator.MAX_UNITS
                or ndims <= 0 or ndims >= RnnValidator.MAX_NDIMS):
            raise InvalidParamsError()


def validate_direction(direction, ndims):
    if not isinstance(direction, Direction):
        raise InvalidParamsError()

    if direction.dimensions != ndims:
        raise InvalidParamsError(direction.dimensions)


class BaseMDRNN(tf.keras.layers.Layer):
    def __init__(self, units, input_shape, kernel_initializer=None,
                 recurrent_initializer=None, bias_initializer=None,
                 return_sequences=False, return_state=False, direction=None, **kwargs):
        if input_shape:
            kwargs.update(dict(input_shape=input_shape))
        super(BaseMDRNN, self).__init__(**kwargs)
        input_dim = input_shape[-1]
        ndims = len(input_shape[:-1])

        RnnValidator.validate_rnn_params(units, input_dim, ndims)

        direction = direction or self._get_default_direction(ndims)
        validate_direction(direction, ndims)

        self._input_shape = input_shape
        self.ndims = ndims
        self.input_dim = input_dim
        self.units = units
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.direction = direction
        self._kernel_initializer = kernel_initializer
        self._recurrent_initializer = recurrent_initializer
        self._bias_initializer = bias_initializer

        self._initialize_initializers()

    def _process_input(self, x_batch, prev_activations, axes):
        raise NotImplementedError

    def _get_default_direction(self, ndims):
        args = [1] * ndims
        return Direction(*args)

    def _initialize_initializers(self):
        default_initializer = self._get_default_initializer()

        self._kernel_initializer = self._kernel_initializer or default_initializer
        self._recurrent_initializer = self._recurrent_initializer or default_initializer
        self._bias_initializer = self._bias_initializer or default_initializer

    def _get_default_initializer(self):
        return tf.keras.initializers.he_normal()

    def call(self, inp, initial_state=None, **kwargs):
        self._validate_input(inp)
        inp = self._prepare_inputs(inp)
        initial_state = self._prepare_initial_state(initial_state)
        outputs = self._make_graph(inp, initial_state)
        return self._prepare_result(outputs)

    def _validate_input(self, inp):
        expected_rank = 1 + self.ndims + 1
        if (len(inp.shape) != expected_rank or 0 in inp.shape
                or inp.shape[-1] != self.input_dim):
            raise InputRankMismatchError(inp.shape)

    def _prepare_inputs(self, inputs):
        if not isinstance(inputs, tf.Tensor):
            inputs = tf.constant(inputs, dtype=tf.float32)

        if inputs.dtype != tf.float32:
            inputs = tf.cast(inputs, dtype=tf.float32)
        return inputs

    def _make_graph(self, inp, initial_state):
        grid_shape = self._calculate_grid_shape(inp)

        tensor_shape = (inp.shape[0], self.input_dim)

        outputs = TensorGrid(grid_shape=grid_shape, tensor_shape=tensor_shape)

        positions = self.direction.iterate_over_positions(grid_shape)

        first_position = positions[0]

        for position in positions:
            x_batch = self._get_batch(inp, position)

            if position == first_position:
                axis_to_activation = [v for v in initial_state]
                axes = list(range(self.ndims))
            else:
                axis_to_activation, axes = self._get_relevant_activations_with_axes(outputs, position)

            a = self._process_input(x_batch, axis_to_activation, axes)
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

    def _get_relevant_activations_with_axes(self, outputs, position):
        previous_positions = self.direction.get_previous_step_positions(position)

        axes_with_positions = self._discard_out_of_bound_positions(
            outputs, previous_positions
        )

        previous_activations = {}
        for axis, prev_position in axes_with_positions:
            previous_activations[axis] = outputs.get_item(prev_position)

        valid_axes = [axis for axis, _ in axes_with_positions]
        return previous_activations, valid_axes

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


class MDRNNCell:
    def __init__(self, kernel_size, input_dim, ndims, kernel_initializer,
                 recurrent_initializer, bias_initializer, activation):

        self.wax = tf.Variable(kernel_initializer((input_dim, kernel_size), dtype=tf.float32))
        self.ba = tf.Variable(bias_initializer((1, kernel_size), dtype=tf.float32))

        self.recurrent_kernels = []
        self._kernel_size = kernel_size
        self._activation = activation

        for _ in range(ndims):
            self.recurrent_kernels.append(
                tf.Variable(recurrent_initializer((self._kernel_size, self._kernel_size), dtype=tf.float32))
            )

    def process(self, x_batch, prev_states, axes):
        z_recurrent = self._compute_weighted_sum_of_activations(prev_states, axes)
        z = tf.add(z_recurrent, tf.matmul(x_batch, self.wax))
        z = tf.add(z, self.ba)
        return self._activation(z)

    def _compute_weighted_sum_of_activations(self, axis_to_activation, axes):
        z_recurrent = tf.zeros((1, self._kernel_size), dtype=tf.float32)
        for axis in axes:
            waa = self.recurrent_kernels[axis]
            a = axis_to_activation[axis]
            z_recurrent = tf.add(tf.matmul(a, waa), z_recurrent)
        return z_recurrent


class MDRNN(BaseMDRNN):
    def __init__(self, units, input_shape, kernel_initializer=None,
                 recurrent_initializer=None, bias_initializer=None, activation='tanh',
                 return_sequences=False, return_state=False, direction=None, **kwargs):
        super(MDRNN, self).__init__(units, input_shape,
                                    kernel_initializer=kernel_initializer,
                                    recurrent_initializer=recurrent_initializer,
                                    bias_initializer=bias_initializer,
                                    return_sequences=return_sequences,
                                    return_state=return_state,
                                    direction=direction,
                                    **kwargs)

        self.activation = tf.keras.activations.get(activation)

        self._cell = MDRNNCell(
            self.units, self.input_dim, self.ndims,
            kernel_initializer=self._kernel_initializer,
            recurrent_initializer=self._recurrent_initializer,
            bias_initializer=self._bias_initializer,
            activation=self.activation
        )

    def spawn(self, direction):
        return MDRNN(units=self.units, input_shape=self._input_shape,
                     kernel_initializer=self._kernel_initializer,
                     recurrent_initializer=self._recurrent_initializer,
                     bias_initializer=self._bias_initializer,
                     activation=self.activation,
                     return_sequences=self.return_sequences,
                     return_state=self.return_state,
                     direction=direction)

    def _process_input(self, x_batch, prev_activations, axes):
        return self._cell.process(x_batch, prev_activations, axes)


class InvalidParamsError(Exception):
    pass


class InputRankMismatchError(Exception):
    pass