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
        pass

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

    def _prepare_initial_state(self, initial_state):
        if initial_state is None:
            return tf.zeros((1, self.units))
        else:
            return initial_state

    def _make_graph(self, inputs, initial_state):
        a = initial_state

        X = inputs

        Tx = X.shape[1]
        tensor_shape = (inputs.shape[0], self.input_dim)

        grid_shape = self._calculate_grid_shape(X)
        output_grid = TensorGrid(grid_shape=grid_shape, tensor_shape=tensor_shape)

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

        return output_grid.to_tensor()
