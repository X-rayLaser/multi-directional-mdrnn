import tensorflow as tf
import numpy as np


class ManyToOneRNN:
    def __init__(self, input_dim, output_size, ndims):
        raise InvalidParamsError()


class MDRNN(tf.keras.layers.Layer):
    MAX_INPUT_DIM = 10**10
    MAX_UNITS = MAX_INPUT_DIM
    MAX_NDIMS = 10**3

    def __init__(self, input_dim, units, ndims, kernel_initializer=None,
                 recurrent_initializer=None, bias_initializer=None, activation='tanh',
                 return_sequences=False, return_state=False, **kwargs):
        super().__init__(**kwargs)
        if (input_dim <= 0 or input_dim >= self.MAX_INPUT_DIM
                or units <= 0 or units >= self.MAX_UNITS
                or ndims <= 0 or ndims >= self.MAX_NDIMS):
            raise InvalidParamsError()

        self.ndims = ndims
        self.input_dim = input_dim
        self.units = units
        self.return_sequences = return_sequences
        self.return_state = return_state

        self.activation = tf.keras.activations.get(activation)

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

    def call(self, inp, initial_state=None):
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

        Tx = inp.shape[1]

        outputs = []

        for i in range(Tx):
            z = tf.add(tf.matmul(a, self.waa), tf.matmul(inp[:, i], self.wax))
            z = tf.add(z, self.ba)
            a = self.activation(z)
            outputs.append(a)

        return tf.stack(outputs, axis=1)

    def _prepare_result(self, outputs):
        last_state = outputs[:, -1]
        if self.return_sequences:
            returned_outputs = outputs
        else:
            returned_outputs = last_state

        if self.return_state:
            return returned_outputs, last_state
        return returned_outputs


class InvalidParamsError(Exception):
    pass


class InputRankMismatchError(Exception):
    pass
