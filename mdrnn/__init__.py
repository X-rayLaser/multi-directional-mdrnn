import tensorflow as tf
import numpy as np


class ManyToOneRNN:
    def __init__(self, input_dim, output_size, ndims):
        raise InvalidParamsError()


class MDRNN:
    MAX_INPUT_DIM = 10**10
    MAX_UNITS = MAX_INPUT_DIM
    MAX_NDIMS = 10**3

    def __init__(self, input_dim, units, ndims, kernel_initializer=None,
                 recurrent_initializer=None, bias_initializer=None, **kwargs):
        if (input_dim <= 0 or input_dim >= self.MAX_INPUT_DIM
                or units <= 0 or units >= self.MAX_UNITS
                or ndims <= 0 or ndims >= self.MAX_NDIMS):
            raise InvalidParamsError()

        self.ndims = ndims
        self.input_dim = input_dim
        self.units = units

        if 'activation' in kwargs:
            self.activation = kwargs['activation']
        else:
            self.activation = tf.tanh

        if kernel_initializer is None:
            self.wax = tf.zeros((units, input_dim), dtype=tf.float32)
        else:
            self.wax = kernel_initializer((units, input_dim), dtype=tf.float32)

        if recurrent_initializer is None:
            self.waa = tf.zeros((units, units), dtype=tf.float32)
        else:
            self.waa = recurrent_initializer((units, units), dtype=tf.float32)

        if bias_initializer is None:
            self.ba = tf.zeros((units, 1), dtype=tf.float32)
        else:
            self.ba = bias_initializer((units, 1), dtype=tf.float32)

    def __call__(self, inp, initial_state=None):
        if (len(inp.shape) != self.ndims + 1 or 0 in inp.shape
                or inp.shape[-1] != self.input_dim):
            raise InputRankMismatchError()

        initial_state = np.zeros((self.units, 1), dtype=np.float)
        a = tf.constant(initial_state, dtype=tf.float32)

        outputs = np.zeros((len(inp), self.units))

        for i in range(len(inp)):
            seq_item = inp[i].reshape(-1, 1)
            x = tf.constant(seq_item, dtype=tf.float32)

            z = tf.add(
                tf.add(
                    tf.matmul(self.waa, a),
                    tf.matmul(self.wax, x)),
                self.ba)

            if self.activation:
                a = tf.tanh(z)
            else:
                a = z
            outputs[i] = a.numpy().ravel()

        return tf.constant(outputs)


class InvalidParamsError(Exception):
    pass


class InputRankMismatchError(Exception):
    pass
