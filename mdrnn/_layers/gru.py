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

    def _prepare_result(self, outputs_grid):
        outputs = outputs_grid.to_tensor()
        final_position = self.direction.get_final_position()
        last_state = self._get_batch(outputs, final_position)

        if self.return_sequences:
            returned_outputs = outputs
        else:
            returned_outputs = last_state

        if self.return_state:
            return [returned_outputs] + [last_state]
        return returned_outputs


class GRUKernelInitializer:
    def __init__(self, initializer, num_kernels):
        pass

    def get_kernel(self, i):
        pass


class GRUBiasInitializer:
    def __init__(self, initializer, num_biases):
        pass

    def get_bias(self, i):
        pass


class GRURecurrentKernelInitializer:
    def __init__(self, initializer, num_md_kernels, num_dimensions):
        pass

    def get_md_kernel(self, i):
        pass

    def get_1d_kernel(self):
        pass


class MDGRUCell:
    def __init__(self, units, input_size, num_dimensions, kernel_initializer_class,
                 recurrent_initializer_class, bias_initializer_class,
                 activation='tanh', recurrent_activation='sigmoid'):
        self._update_gates = []
        self._relevance_gates = []
        self._num_dimensions = num_dimensions

        num_gate_kernels = 2 * num_dimensions
        num_state_kernels = 1
        num_kernels = num_gate_kernels + num_state_kernels
        kernel_initializer = GRUKernelInitializer(kernel_initializer_class, num_kernels, input_size, units)

        for i in range(num_dimensions):
            kernel_initializer = kernel_initializer_class(seed=i)
            recurrent_initializer = recurrent_initializer_class(seed=i)
            bias_initializer = bias_initializer_class(seed=i)
            gate = LinearGate(units, input_size, num_dimensions,
                              kernel_initializer,
                              recurrent_initializer, bias_initializer)
            self._update_gates.append(gate)

            kernel_initializer = kernel_initializer_class(seed=i + num_dimensions)
            recurrent_initializer = recurrent_initializer_class(seed=i + num_dimensions)
            bias_initializer = bias_initializer_class(seed=i + num_dimensions)

            gate = LinearGate(units, input_size, num_dimensions,
                              kernel_initializer,
                              recurrent_initializer, bias_initializer)
            self._relevance_gates.append(gate)

        kernel_initializer = kernel_initializer_class(seed=num_dimensions * 2)
        recurrent_initializer = recurrent_initializer_class(seed=num_dimensions * 2)
        bias_initializer = bias_initializer_class(seed=num_dimensions * 2)

        self.state = LinearGate(units, input_size, num_dimensions, kernel_initializer,
                                recurrent_initializer, bias_initializer)

        self._activation = tf.keras.activations.get(activation)
        self._recurrent_activation = tf.keras.activations.get(recurrent_activation)

    def process(self, x_batch, prev_states, axes):
        updates = self._gated_computation(x_batch, prev_states, axes,
                                          self._update_gates)
        relevancy = self._gated_computation(x_batch, prev_states, axes,
                                            self._relevance_gates)

        gated_states = [relevancy[d] * prev_states[d] for d in axes]

        z = self._activation(self.state.process(x_batch, gated_states, axes))

        output = tf.zeros_like(updates[0])
        for d in axes:
            output += updates[d] * z + (1 - updates[d]) * prev_states[d]

        return output

    def _gated_computation(self, x_batch, prev_states, axes, gates):
        res = []

        for d in range(self._num_dimensions):
            z = gates[d].process(x_batch, prev_states, axes)
            output = self._recurrent_activation(z)
            res.append(output)
        return res


class LinearGate:
    MAX_NUM_DIMENSIONS = 1000

    def __init__(self, units, input_size, num_dimensions, kernel_initializer,
                 recurrent_initializer, bias_initializer):
        if num_dimensions < 0 or num_dimensions > self.MAX_NUM_DIMENSIONS:
            raise BadDimensionalityError()

        self.units = units
        self.num_dimensions = num_dimensions
        self.kernel = tf.Variable(kernel_initializer((input_size, units), dtype=tf.float64))

        self.aggregate_kernel = tf.Variable(recurrent_initializer((units, units * num_dimensions), dtype=tf.float64))
        self.recurrent_kernels = []
        self.set_recurrent_kernel(self.aggregate_kernel)

        self.bias = tf.Variable(bias_initializer((units,), dtype=tf.float64))

    def set_kernel(self, kernel):
        self.kernel = kernel

    def set_recurrent_kernel(self, aggregate_kernel):
        self.recurrent_kernels = []
        for i in range(self.num_dimensions):
            k = aggregate_kernel[:, i * self.units:(i + 1) * self.units]
            self.recurrent_kernels.append(k)

    def set_bias(self, bias):
        self.bias = bias

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
