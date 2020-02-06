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
            self.wax = default_initializer((units, input_dim), dtype=tf.float32)
        else:
            self.wax = kernel_initializer((units, input_dim), dtype=tf.float32)

        if recurrent_initializer is None:
            self.waa = default_initializer((units, units), dtype=tf.float32)
        else:
            self.waa = recurrent_initializer((units, units), dtype=tf.float32)

        if bias_initializer is None:
            self.ba = default_initializer((units, 1), dtype=tf.float32)
        else:
            self.ba = bias_initializer((units, 1), dtype=tf.float32)

    def call(self, inp, initial_state=None):
        self._validate_input(inp)

        if initial_state is None:
            initial_state = np.zeros((self.units, 1), dtype=np.float)
        a = tf.constant(initial_state, dtype=tf.float32)

        Tx = inp.shape[1]

        outputs = np.zeros((1, Tx, self.units))

        if inp.shape[0] is not None and inp.shape[0] > 1:
            return np.zeros((inp.shape[0], Tx, self.units))

        inp = tf.constant(inp, dtype=tf.float32)
        for i in range(Tx):
            seq_item = tf.reshape(inp[0][i], (-1, 1))
            x = tf.constant(seq_item, dtype=tf.float32)

            z = tf.add(
                tf.add(
                    tf.matmul(self.waa, a),
                    tf.matmul(self.wax, x)),
                self.ba)

            a = self.activation(z)
            outputs[0, i] = a.numpy().ravel()

        return self._prepare_result(outputs)

    def _validate_input(self, inp):
        expected_rank = 1 + self.ndims + 1
        if (len(inp.shape) != expected_rank or 0 in inp.shape
                or inp.shape[-1] != self.input_dim):
            raise InputRankMismatchError(inp.shape)

    def _prepare_result(self, outputs):
        last_state = tf.constant(outputs[:, -1])
        if self.return_sequences:
            returned_outputs = tf.constant(outputs)
        else:
            returned_outputs = last_state

        if self.return_state:
            return returned_outputs, last_state
        return returned_outputs


class InvalidParamsError(Exception):
    pass


class InputRankMismatchError(Exception):
    pass
