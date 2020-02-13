import tensorflow as tf
import numpy as np
import itertools


class ManyToOneRNN:
    def __init__(self, input_dim, output_size, ndims):
        raise InvalidParamsError()


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

        if initial_state is None:
            initial_state = np.zeros((1, self.units), dtype=np.float)

        outputs = self._make_graph(inp, initial_state)

        return self._prepare_result(outputs)

    def _validate_input(self, inp):
        expected_rank = 1 + self.ndims + 1
        if (len(inp.shape) != expected_rank or 0 in inp.shape
                or inp.shape[-1] != self.input_dim):
            raise InputRankMismatchError(inp.shape)

    def _make_graph(self, inp, initial_state):
        a = tf.constant(initial_state, dtype=tf.float32)

        dim_lengths = inp.shape[1:-1]

        tensor_shape = (inp.shape[0], self.input_dim)
        print(tensor_shape)
        outputs = TensorGrid(grid_shape=dim_lengths, tensor_shape=tensor_shape)

        positions = self.direction.iterate_positions(dim_lengths)

        for position in positions:
            batch = self._get_batch(inp, position)
            z_recurrent = tf.zeros((1, self.units), dtype=tf.float32)

            if self.ndims == 1:
                waa = self.recurrent_kernels[0]
                z_recurrent = tf.add(tf.matmul(a, waa), z_recurrent)
            else:
                for d in range(len(self.recurrent_kernels)):
                    waa = self.recurrent_kernels[d]
                    prev_position = self._prev_position(position, d)

                    bad_case = False
                    for index in prev_position:
                        if index < 0:
                            bad_case = True

                    if bad_case:
                        continue
                    a = outputs.get_item(prev_position)
                    z_recurrent = tf.add(tf.matmul(a, waa), z_recurrent)

            z = tf.add(z_recurrent, tf.matmul(batch, self.wax))
            z = tf.add(z, self.ba)
            a = self.activation(z)

            outputs.put_item(position, a)

        return outputs.to_tensor()

    def _get_batch(self, tensor, position):
        i = position[0]
        t = tensor[:, i]

        for index in position[1:]:
            t = t[:, index]
        return t

    def _prev_position(self, position, d):
        if self.ndims == 1:
            return position[0] - 1,

        if self.ndims == 2:
            i, j = tuple(position)
            if d == 0:
                return i - 1, j
            else:
                return i, j - 1

    def _prepare_result(self, outputs):
        last_state = outputs[:, -1]
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

    @property
    def dimensions(self):
        return len(self._dirs)

    def iterate_positions(self, dim_lengths):
        axes = []

        for i, dim_len in enumerate(dim_lengths):
            step_size = self._dirs[i]
            if step_size > 0:
                index_iter = range(dim_len)
            else:
                index_iter = range(dim_len - 1, -1, -1)

            axes.append(index_iter)

        return list(itertools.product(*axes))


class InvalidParamsError(Exception):
    pass


class InputRankMismatchError(Exception):
    pass
