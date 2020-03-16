import tensorflow as tf
from mdrnn._util.grids import TensorGrid
from mdrnn._layers.simple_mdrnn import BaseMDRNN


class MDGRU(BaseMDRNN):
    def __init__(self, units, input_shape, kernel_initializer=None,
                 recurrent_initializer=None, bias_initializer=None, activation='tanh',
                 recurrent_activation='sigmoid',
                 return_sequences=False, return_state=False, direction=None, **kwargs):
        super(MDGRU, self).__init__(units, input_shape,
                                    kernel_initializer=kernel_initializer,
                                    recurrent_initializer=recurrent_initializer,
                                    bias_initializer=bias_initializer,
                                    return_sequences=return_sequences,
                                    return_state=return_state,
                                    direction=direction,
                                    **kwargs)

        self.activation = tf.keras.activations.get(activation)
        self.recurrent_activation = tf.keras.activations.get(recurrent_activation)

        self._initialize_weights()

    def _process_input(self, x_batch, prev_activations, axes):
        # todo: implement this method using custom Cell class
        a = prev_activations[axes[0]]
        term = tf.matmul(x_batch, self.Wz) + self.Bz
        z = self.recurrent_activation(tf.matmul(a, self.Uz) + term)

        term = tf.matmul(x_batch, self.Wr) + self.Br
        r = self.recurrent_activation(tf.matmul(a, self.Ur) + term)

        term = tf.matmul(x_batch, self.Wh) + self.Bh
        h = self.activation(tf.matmul(r * a, self.Uh) + term)

        a = z * a + (1 - z) * h
        return a

    def _initialize_weights(self):
        input_size = self._input_shape[-1]
        self.kernel = self.add_weight(
            shape=(input_size, self.units * 3),
            name='kernel',
            initializer=self._kernel_initializer)

        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 3),
            name='recurrent_kernel',
            initializer=self._recurrent_initializer)

        self.bias = self.add_weight(shape=(self.units * 3, ),
                                    name='bias',
                                    initializer=self._bias_initializer)

        self.Wz = self.kernel[:, :self.units]
        self.Wr = self.kernel[:, self.units: self.units * 2]
        self.Wh = self.kernel[:, self.units * 2:]

        self.Uz = self.recurrent_kernel[:, :self.units]
        self.Ur = self.recurrent_kernel[:, self.units: self.units * 2]
        self.Uh = self.recurrent_kernel[:, self.units * 2:]

        self.Bz = self.bias[:self.units]
        self.Br = self.bias[self.units: self.units * 2]
        self.Bh = self.bias[self.units * 2:]

    def spawn(self, direction):
        return MDGRU(units=self.units, input_shape=self._input_shape,
                     kernel_initializer=self._kernel_initializer,
                     recurrent_initializer=self._recurrent_initializer,
                     bias_initializer=self._bias_initializer,
                     activation=self.activation,
                     recurrent_activation=self.recurrent_activation,
                     return_sequences=self.return_sequences,
                     return_state=self.return_state,
                     direction=direction)


class LinearGate:
    MAX_NUM_DIMENSIONS = 1000

    def __init__(self, units, input_size, num_dimensions, kernel_initializer,
                 recurrent_initializer, bias_initializer):
        if num_dimensions < 0 or num_dimensions > self.MAX_NUM_DIMENSIONS:
            raise BadDimensionalityError()

        self.num_dimensions = num_dimensions
        self.kernel = tf.Variable(kernel_initializer((input_size, units), dtype=tf.float64))

        self.aggregate_kernel = tf.Variable(recurrent_initializer((units, units * num_dimensions), dtype=tf.float64))

        self.recurrent_kernels = []
        for i in range(num_dimensions):
            k = self.aggregate_kernel[:, i * units:(i + 1) * units]
            self.recurrent_kernels.append(k)

        self.bias = tf.Variable(bias_initializer((units,), dtype=tf.float64))

    def process(self, x_batch, prev_outputs, axes):
        if len(prev_outputs) > self.num_dimensions or len(axes) > self.num_dimensions:
            raise Exception('# of elements in a list of previous outputs '
                            'greater than number of input dimensions')

        output = tf.matmul(x_batch, self.kernel)

        recurrent_sum = tf.zeros_like(output)

        for axis in axes:
            a = prev_outputs[axis]
            recurrent_sum += tf.matmul(a, self.recurrent_kernels[axis])

        return output + recurrent_sum + self.bias


class BadDimensionalityError(Exception):
    pass
