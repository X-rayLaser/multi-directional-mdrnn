import tensorflow as tf
import numpy as np
from .simple_mdrnn import BaseMDRNN


class MDLSTM(BaseMDRNN):
    def spawn(self, direction):
        return MDLSTM(units=self.units, input_shape=self._input_shape,
                      kernel_initializer=self._kernel_initializer,
                      recurrent_initializer=self._recurrent_initializer,
                      bias_initializer=self._bias_initializer,
                      activation=self.activation,
                      return_sequences=self.return_sequences,
                      return_state=self.return_state,
                      direction=direction)

    def _prepare_initial_state(self, initial_state):
        a0 = super()._prepare_initial_state(initial_state)
        c0 = super()._prepare_initial_state(initial_state)
        return a0, c0

    def _process_input(self, x_batch, prev_activations, axes):
        pass

    def _prepare_result(self, outputs_grid):
        pass


class MDLSTMCell:
    def __init__(self, kernel_size, input_dim, ndims, kernel_initializer,
                 recurrent_initializer, shared_weight_initializer, activation):
        self.input_gate = Gate(kernel_size, input_dim, ndims,
                               kernel_initializer, recurrent_initializer,
                               shared_weight_initializer, activation)
        self.forget_gates = []
        for _ in range(ndims):
            gate = Gate(kernel_size, input_dim, ndims,
                        kernel_initializer, recurrent_initializer,
                        shared_weight_initializer, activation)
            self.forget_gates.append(gate)

        self.output_gate = Gate(kernel_size, input_dim, ndims,
                                kernel_initializer, recurrent_initializer,
                                shared_weight_initializer, activation)

    def process(self, x_batch, prev_states, axes):
        b_input = self.input_gate.compute(x_batch, prev_states, axes)

        b_forget = [0] * len(self.forget_gates)

        for d in axes:
            b_forget[d] = self.forget_gates[d].compute(x_batch, prev_states, axes)

        b_output = self.output_gate.compute(x_batch, prev_states, axes)

        # todo: compute cell states and outputs

    def _compute_input_gate(self, x_batch, b, s):
        pass

    def _compute_forget_gate(self, x_batch, b, s, d):
        pass

    def _compute_cell(self, x_batch, b, b_i, s, b_phi):
        pass

    def _compute_output(self, s, b):
        pass


class Gate:
    def __init__(self, units, input_dim, ndims, kernel_initializer,
                 recurrent_initializer, shared_weight_initializer, activation):
        self.units = units
        self.kernel = tf.Variable(kernel_initializer((input_dim, units), dtype=tf.float32))

        self.recurrent_kernels = []
        for _ in range(ndims):
            w = tf.Variable(recurrent_initializer((units, units), dtype=tf.float32))
            self.recurrent_kernels.append(w)

        self.shared_kernel = tf.Variable(shared_weight_initializer((units,), dtype=tf.float32))
        self.activation = activation

    def compute(self, x_batch, prev_states, axes):
        z_recurrent = tf.zeros((1, self.units), dtype=tf.float32)

        for axis in axes:
            b, s = prev_states[axis]
            z_recurrent = tf.add(tf.matmul(b, self.recurrent_kernels[axis]), z_recurrent)
            z_recurrent = tf.add(tf.matmul(s, self.shared_kernel), z_recurrent)

        a = tf.add(tf.matmul(x_batch, self.kernel), z_recurrent)
        return self.activation(a)


class CellState:
    def __init__(self, units, input_dim, ndims, kernel_initializer,
                 recurrent_initializer, activation):
        self.units = units
        self.kernel = tf.Variable(kernel_initializer((input_dim, units), dtype=tf.float32))

        self.recurrent_kernels = []
        for _ in range(ndims):
            w = tf.Variable(recurrent_initializer((units, units), dtype=tf.float32))
            self.recurrent_kernels.append(w)

        self.activation = activation

    def compute(self, x_batch, prev_states, axes, b_input, b_forget):
        z_recurrent = tf.zeros((1, self.units), dtype=tf.float32)

        for axis in axes:
            b, s = prev_states[axis]
            z_recurrent = tf.add(tf.matmul(b, self.recurrent_kernels[axis]), z_recurrent)

        a = tf.add(tf.matmul(x_batch, self.kernel), z_recurrent)

        tf.add(tf.multiply(b_input, self.activation(a)),
               )
        return self.activation(a)

    def _sum_over_dimensions(self, cell_states, b_forget, axes):
        # todo: implement this
        pass