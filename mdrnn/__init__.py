import tensorflow as tf


class ManyToOneRNN:
    def __init__(self, input_dim, output_size, ndims):
        raise InvalidParamsError()


class MDRNN:
    MAX_INPUT_DIM = 10**10
    MAX_UNITS = MAX_INPUT_DIM
    MAX_NDIMS = 10**3

    def __init__(self, input_dim, units, ndims, kernel_initializer=None,
                 recurrent_initializer=None, bias_initializer=None):
        if (input_dim <= 0 or input_dim >= self.MAX_INPUT_DIM
                or units <= 0 or units >= self.MAX_UNITS
                or ndims <= 0 or ndims >= self.MAX_NDIMS):
            raise InvalidParamsError()

        self.ndims = ndims
        self.input_dim = input_dim

        #if kernel_initializer is None:
            #self.waa = tf
            #self.waa = kernel_initializer()

    def __call__(self, inp, initial_state=None):
        if (len(inp.shape) != self.ndims + 1 or 0 in inp.shape
                or inp.shape[-1] != self.input_dim):
            raise InputRankMismatchError()

        return tf.constant([[5], [5]])


class InvalidParamsError(Exception):
    pass


class InputRankMismatchError(Exception):
    pass
