from tensorflow.keras.layers import Layer
import tensorflow as tf

from mdrnn._layers.simple_mdrnn import InputRankMismatchError, InvalidParamsError
from mdrnn._util.directions import Direction


class MDGRU(Layer):
    MAX_INPUT_DIM = 10**10
    MAX_UNITS = MAX_INPUT_DIM
    MAX_NDIMS = 10**3

    def __init__(self, units, input_shape, kernel_initializer=None,
                 recurrent_initializer=None, bias_initializer=None, activation='tanh',
                 recurrent_activation='sigmoid',
                 return_sequences=False, return_state=False, direction=None, **kwargs):
        super(MDGRU, self).__init__(**kwargs)

        input_dim = input_shape[-1]
        ndims = len(input_shape[:-1])

        if (input_dim <= 0 or input_dim >= self.MAX_INPUT_DIM
                or units <= 0 or units >= self.MAX_UNITS
                or ndims <= 0 or ndims >= self.MAX_NDIMS):
            raise InvalidParamsError()

        if direction is None:
            args = [1] * ndims
            direction = Direction(*args)

        self.input_dim = input_dim
        self.ndims = ndims
        self.units = units
        self.direction = direction

        input_size = input_shape[-1]
        self.kernel = self.add_weight(
            shape=(input_size, self.units * 3),
            name='kernel',
            initializer=kernel_initializer)

        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 3),
            name='recurrent_kernel',
            initializer=recurrent_initializer)

        self.bias = self.add_weight(shape=(units * 3, ),
                                    name='bias',
                                    initializer=bias_initializer)

        self.Wz = self.kernel[:, :self.units]
        self.Wr = self.kernel[:, self.units: self.units * 2]
        self.Wh = self.kernel[:, self.units * 2:]

        self.Uz = self.recurrent_kernel[:, :self.units]
        self.Ur = self.recurrent_kernel[:, self.units: self.units * 2]
        self.Uh = self.recurrent_kernel[:, self.units * 2:]

        self.Bz = self.bias[:self.units]
        self.Br = self.bias[self.units: self.units * 2]
        self.Bh = self.bias[self.units * 2:]

        self.activation = tf.keras.activations.get(activation)
        self.recurrent_activation = tf.keras.activations.get(recurrent_activation)

    def call(self, inputs, initial_state=None, **kwargs):
        self._validate_input(inputs)

        a = tf.zeros((1, self.units))
        X = tf.constant(inputs, dtype=tf.float32)

        Tx = X.shape[1]

        for position in self.direction.iterate_over_positions([Tx]):
            t = position[0]
            x = X[:, t, :]
            term = tf.matmul(x, self.Wz) + self.Bz
            z = self.recurrent_activation(tf.matmul(a, self.Uz) + term)

            term = tf.matmul(x, self.Wr) + self.Br
            r = self.recurrent_activation(tf.matmul(a, self.Ur) + term)

            term = tf.matmul(x, self.Wh) + self.Bh
            h = self.activation(tf.matmul(r * a, self.Uh) + term)

            a = z * a + (1 - z) * h
        return a

    def _validate_input(self, inp):
        expected_rank = 1 + self.ndims + 1
        if (len(inp.shape) != expected_rank or 0 in inp.shape
                or inp.shape[-1] != self.input_dim):
            raise InputRankMismatchError(inp.shape)