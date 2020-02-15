import tensorflow as tf
import numpy as np
import itertools


class MDRNN(tf.keras.layers.Layer):
    MAX_INPUT_DIM = 10**10
    MAX_UNITS = MAX_INPUT_DIM
    MAX_NDIMS = 10**3

    def __init__(self, units, input_shape, kernel_initializer=None,
                 recurrent_initializer=None, bias_initializer=None, activation='tanh',
                 return_sequences=False, return_state=False, direction=None, **kwargs):
        if input_shape:
            kwargs.update(dict(input_shape=input_shape))
        super().__init__(**kwargs)

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
            return returned_outputs, last_state
        return returned_outputs


class MultiDirectional(tf.keras.layers.Layer):
    def __init__(self, rnn, **kwargs):
        super().__init__(**kwargs)
        self._forward_rnn = rnn.spawn(direction=Direction(1))
        self._backward_rnn = rnn.spawn(direction=Direction(-1))

    def build(self, input_shape):
        pass

    def call(self, inputs, **kwargs):
        a_forward = self._forward_rnn.call(inputs, **kwargs)
        a_backward = self._backward_rnn.call(inputs, **kwargs)

        return tf.concat([a_forward, a_backward], axis=2)


class MultiDimensionalGrid:
    def __init__(self, grid_shape):
        self._shape = grid_shape
        self._grid = np.zeros(grid_shape).tolist()

    def get_positions(self):
        iterators = []
        for dim in self._shape:
            iterators.append(range(dim))

        for position in itertools.product(*iterators):
            yield position

    def set_grid(self, grid):
        self._grid = list(grid)

    def put_item(self, position, item):
        inner_list = self._get_inner_most_list(position)
        inner_list[position[-1]] = item

    def get_item(self, position):
        if len(position) != len(self._shape):
            raise InvalidPositionError()

        self._validate_position_dimensions(position)

        row = self._get_inner_most_list(position)
        return row[position[-1]]

    def get_sub_list(self, position):
        self._validate_position_dimensions(position)

        row = self._get_inner_most_list(position)
        return row[position[-1]]

    def _validate_position_dimensions(self, position):
        for i, xd in enumerate(position):
            if xd < 0 or xd >= self._shape[i]:
                raise PositionOutOfBoundsError()

    def _get_inner_most_list(self, position):
        if len(position) > 1:
            sub_list = self._grid[position[0]]
            for index in position[1:-1]:
                sub_list = sub_list[index]

            return sub_list
        else:
            return self._grid


class TensorGrid(MultiDimensionalGrid):
    def __init__(self, grid_shape, tensor_shape):
        super().__init__(grid_shape)

        for position in self.get_positions():
            if tensor_shape[0] is None:
                tensor_shape = (1,) + tensor_shape[1:]
            zeros_tensor = tf.zeros(shape=tensor_shape)
            self.put_item(position, zeros_tensor)

        self._tensor_shape = tensor_shape

    @property
    def grid_shape(self):
        return self._shape

    def reduce_rank(self):
        if len(self.grid_shape) == 1:
            tensor = tf.stack(self._grid, axis=1)
            return NullGrid(tensor)

        new_tensor_shape = self._calculate_new_tensor_shape()
        new_grid_shape = self._calculate_new_grid_shape()

        new_grid = TensorGrid(grid_shape=new_grid_shape,
                              tensor_shape=new_tensor_shape)

        for position in new_grid.get_positions():
            tensor = self._fold_sub_list(position)
            new_grid.put_item(position, tensor)

        return new_grid

    def _calculate_new_tensor_shape(self):
        batch_size = self._tensor_shape[0]
        new_tensor_shape = (batch_size,) + (self.grid_shape[-2],) + self._tensor_shape[1:]
        return new_tensor_shape

    def _calculate_new_grid_shape(self):
        return self.grid_shape[:-1]

    def _fold_sub_list(self, position):
        inner_list = self.get_sub_list(position)
        return tf.stack(inner_list, axis=1)

    def to_tensor(self):
        new_grid = self.reduce_rank()

        while len(new_grid.grid_shape) > 0:
            new_grid = new_grid.reduce_rank()

        return new_grid.get_item((0,))


class NullGrid(MultiDimensionalGrid):
    def __init__(self, tensor):
        super().__init__(tuple())
        self._tensor = tensor

    @property
    def grid_shape(self):
        return tuple()

    def get_item(self, position):
        return self._tensor


class PositionOutOfBoundsError(Exception):
    pass


class InvalidPositionError(Exception):
    pass


class Direction:
    def __init__(self, *directions):
        self._dirs = list(directions)
        for d in self._dirs:
            if d not in [-1, 1]:
                raise TypeError('data type not understood')

    @property
    def dimensions(self):
        return len(self._dirs)

    def iterate_over_positions(self, dim_lengths):
        axes = []

        for i, dim_len in enumerate(dim_lengths):
            step_size = self._dirs[i]
            if step_size > 0:
                index_iter = range(dim_len)
            else:
                index_iter = range(dim_len - 1, -1, -1)

            axes.append(index_iter)

        return list(itertools.product(*axes))

    def get_previous_step_positions(self, current_position):
        res = []
        for d in range(len(self._dirs)):
            position = self._get_previous_position_along_axis(current_position, d)
            res.append(position)

        return res

    def get_final_position(self):
        position = []
        for d in self._dirs:
            if d == 1:
                position.append(-1)
            elif d == -1:
                position.append(0)
            else:
                raise Exception('Should not get here')
        return tuple(position)

    def _get_previous_position_along_axis(self, position, axis):
        step = self._dirs[axis]
        return position[:axis] + (position[axis] - step,) + position[axis + 1:]


class InvalidParamsError(Exception):
    pass


class InputRankMismatchError(Exception):
    pass
