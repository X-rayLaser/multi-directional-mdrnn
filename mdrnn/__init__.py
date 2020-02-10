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
            self.wax = default_initializer((input_dim, units), dtype=tf.float32)
        else:
            self.wax = kernel_initializer((input_dim, units), dtype=tf.float32)

        if recurrent_initializer is None:
            self.waa = default_initializer((units, units), dtype=tf.float32)
        else:
            self.waa = recurrent_initializer((units, units), dtype=tf.float32)

        if bias_initializer is None:
            self.ba = default_initializer((1, units), dtype=tf.float32)
        else:
            self.ba = bias_initializer((1, units), dtype=tf.float32)

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

        outputs = np.zeros(inp.shape[1:-1], dtype=np.int).tolist()

        positions = self.direction.iterate_positions(dim_lengths)

        for position in positions:
            batch = self._get_batch(inp, position)
            z = tf.add(tf.matmul(a, self.waa), tf.matmul(batch, self.wax))
            z = tf.add(z, self.ba)
            a = self.activation(z)

            self._add_result(outputs, position, a)

        res = tf.stack(outputs, axis=self.ndims)
        return res

    def _get_batch(self, tensor, position):
        i = position[0]
        t = tensor[:, i]

        for index in position[1:]:
            t = t[:, index]
        return t

    def _add_result(self, outputs, position, a):
        if len(position) > 1:
            sub_list = outputs[position[0]]
            for index in position[1:-1]:
                sub_list = sub_list[index]

            sub_list[position[-1]] = a
        else:
            outputs[position[0]] = a

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
