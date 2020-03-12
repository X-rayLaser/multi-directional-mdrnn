from tensorflow.keras.layers import Layer
import tensorflow as tf

from mdrnn._layers.simple_mdrnn import InputRankMismatchError, InvalidParamsError
from mdrnn._util.directions import Direction
from mdrnn._util.grids import TensorGrid
from mdrnn._layers.simple_mdrnn import RnnValidator, BaseMDRNN


class MDGRU(BaseMDRNN):
    def __init__(self, units, input_shape, kernel_initializer=None,
                 recurrent_initializer=None, bias_initializer=None, activation='tanh',
                 recurrent_activation='sigmoid',
                 return_sequences=False, return_state=False, direction=None, **kwargs):
        super(MDGRU, self).__init__(units, input_shape,
                                    kernel_initializer=kernel_initializer,
                                    recurrent_initializer=recurrent_initializer,
                                    bias_initializer=bias_initializer,
                                    activation=activation,
                                    return_sequences=return_sequences,
                                    return_state=return_state,
                                    direction=direction,
                                    **kwargs)

        self.recurrent_activation = tf.keras.activations.get(recurrent_activation)

        self._initialize_weights()

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

    def call(self, inputs, initial_state=None, **kwargs):
        self._validate_input(inputs)

        a = tf.zeros((1, self.units))
        X = tf.constant(inputs, dtype=tf.float32)

        Tx = X.shape[1]
        tensor_shape = (inputs.shape[0], self.input_dim)

        output_grid = TensorGrid(grid_shape=(Tx, ), tensor_shape=tensor_shape)

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
            output_grid.put_item(position, a)

        outputs = output_grid.to_tensor()

        last_state = a

        if self.return_sequences:
            returned_outputs = outputs
        else:
            returned_outputs = last_state

        if self.return_state:
            return [returned_outputs] + [last_state]
        return returned_outputs
