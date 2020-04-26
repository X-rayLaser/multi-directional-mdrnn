import tensorflow as tf
from .simple_mdrnn import BaseMDRNN
from .._util.grids import LSTMCellGrid


class MDLSTM(BaseMDRNN):
    def __init__(self, units, input_shape, kernel_initializer=None,
                 recurrent_initializer=None, peephole_initializer=None,
                 gate_activation='sigmoid', cell_input_activation='tanh',
                 cell_output_activation='tanh',
                 return_sequences=False, return_state=False, direction=None,
                 **kwargs):
        super(MDLSTM, self).__init__(units, input_shape,
                                     kernel_initializer=kernel_initializer,
                                     recurrent_initializer=recurrent_initializer,
                                     return_sequences=return_sequences,
                                     return_state=return_state,
                                     direction=direction,
                                     **kwargs)

        default_initializer = self._get_default_initializer()
        self._peephole_initializer = peephole_initializer or default_initializer

        self._gate_activation = tf.keras.activations.get(gate_activation)
        self._cell_input_activation = tf.keras.activations.get(cell_input_activation)
        self._cell_output_activation = tf.keras.activations.get(cell_output_activation)

        self._cell = MDLSTMCell(
            self.units, self.input_dim, self.ndims,
            kernel_initializer=self._kernel_initializer,
            recurrent_initializer=self._recurrent_initializer,
            peephole_initializer=self._peephole_initializer,
            gate_activation=self._gate_activation,
            cell_input_activation=self._cell_input_activation,
            cell_output_activation=self._cell_output_activation
        )

    def spawn(self, direction):
        return MDLSTM(units=self.units, input_shape=self._input_shape,
                      kernel_initializer=self._kernel_initializer,
                      recurrent_initializer=self._recurrent_initializer,
                      peephole_initializer=self._peephole_initializer,
                      gate_activation=self._gate_activation,
                      cell_input_activation=self._cell_input_activation,
                      cell_output_activation=self._cell_output_activation,
                      return_sequences=self.return_sequences,
                      return_state=self.return_state,
                      direction=direction)

    def _create_grid(self, grid_shape, tensor_shape):
        return LSTMCellGrid(grid_shape, tensor_shape)

    def _prepare_initial_state(self, initial_state):
        a0 = super(MDLSTM, self)._prepare_initial_state(initial_state)
        c0 = super(MDLSTM, self)._prepare_initial_state(initial_state)

        return list(zip(a0, c0))

    def _process_input(self, x_batch, prev_activations, axes):
        return self._cell.process(x_batch, prev_activations, axes)

    def _prepare_result(self, outputs_grid):
        c = outputs_grid.get_cells_tensor()
        a = outputs_grid.get_activations_tensor()

        final_position = self.direction.get_final_position()
        last_state_a = self._get_batch(a, final_position)
        last_state_c = self._get_batch(c, final_position)

        if self.return_sequences:
            returned_outputs = a
        else:
            returned_outputs = last_state_a

        if self.return_state:
            return [returned_outputs] + [last_state_a] + [last_state_c]
        return returned_outputs


class MDLSTMCell:
    def __init__(self, units, input_dim, ndims, kernel_initializer,
                 recurrent_initializer, peephole_initializer,
                 gate_activation, cell_input_activation, cell_output_activation):
        self.input_gate = InputGate(units, input_dim, ndims,
                                    kernel_initializer, recurrent_initializer,
                                    peephole_initializer, gate_activation)
        self.forget_gates = []
        for axis in range(ndims):
            gate = ForgetGate(units, input_dim, ndims,
                              kernel_initializer, recurrent_initializer,
                              peephole_initializer, gate_activation)
            self.forget_gates.append(gate)

        self.output_gate = OutputGate(units, input_dim, ndims,
                                      kernel_initializer, recurrent_initializer,
                                      peephole_initializer, gate_activation)

        self.cell_state = CellState(units, input_dim, ndims,
                                    kernel_initializer,
                                    recurrent_initializer, cell_input_activation)

        self.cell_output_activation = cell_output_activation

    def process(self, x_batch, prev_states, axes):
        b_input = self.input_gate.compute(x_batch, prev_states, axes)

        b_forget = [0] * len(self.forget_gates)

        for d in axes:
            b_forget[d] = self.forget_gates[d].compute(x_batch, prev_states, axes, axis=d)

        state = self.cell_state.compute(x_batch, prev_states, axes,
                                        b_input=b_input, b_forget=b_forget)

        b_output = self.output_gate.compute(x_batch, prev_states, axes, cells=state)

        return tf.multiply(self.cell_output_activation(state), b_output)


class ComputationGraph(object):
    def __init__(self, units, input_dim, ndims, kernel_initializer, recurrent_initializer):
        self.units = units
        self.kernel = tf.Variable(kernel_initializer((input_dim, units), dtype=tf.float32))

        self.recurrent_kernels = []
        for _ in range(ndims):
            w = tf.Variable(recurrent_initializer((units, units), dtype=tf.float32))
            self.recurrent_kernels.append(w)

    def _compute_inner_products_sum(self, prev_states, axes):
        z = tf.zeros((1, self.units), dtype=tf.float32)

        for axis in axes:
            b, s = prev_states[axis]
            z = tf.add(tf.matmul(b, self.recurrent_kernels[axis]), z)

        return z

    def _multiply_by_kernel(self, x_batch):
        return tf.matmul(x_batch, self.kernel)

    def _compute_hadamard_product_sum(self, v, w, axes):
        z = tf.zeros((1, self.units), dtype=tf.float32)

        for axis in axes:
            z = tf.add(tf.multiply(v[axis], w[axis]), z)

        return z

    def compute(self, x_batch, prev_states, axes, **kwargs):
        product_sum = self._compute_inner_products_sum(prev_states, axes)
        return tf.add(self._multiply_by_kernel(x_batch), product_sum)


class InputGate(ComputationGraph):
    def __init__(self, units, input_dim, ndims, kernel_initializer,
                 recurrent_initializer, peephole_initializer, activation):
        super(InputGate, self).__init__(units, input_dim, ndims, kernel_initializer, recurrent_initializer)
        self.units = units
        self.kernel = tf.Variable(kernel_initializer((input_dim, units), dtype=tf.float32))

        self.recurrent_kernels = []
        for _ in range(ndims):
            w = tf.Variable(recurrent_initializer((units, units), dtype=tf.float32))
            self.recurrent_kernels.append(w)

        self.shared_kernel = tf.Variable(peephole_initializer((units,), dtype=tf.float32))
        self.activation = activation

    def compute(self, x_batch, prev_states, axes, **kwargs):
        a = super(InputGate, self).compute(x_batch, prev_states, axes, **kwargs)
        extra_term = self._compute_extra_term(prev_states, axes, **kwargs)
        a = tf.add(a, extra_term)
        return self.activation(a)

    def _compute_extra_term(self, prev_states, axes, **kwargs):
        axis_to_cell = {}
        axis_to_kernel = {}

        for axis in axes:
            b, s = prev_states[axis]
            axis_to_cell[axis] = s
            axis_to_kernel[axis] = self.shared_kernel

        return self._compute_hadamard_product_sum(
            axis_to_cell, axis_to_kernel, axes
        )


class ForgetGate(InputGate):
    def _compute_extra_term(self, prev_states, axes, **kwargs):
        axis = kwargs['axis']
        b, s = prev_states[axis]
        return tf.multiply(s, self.shared_kernel)


class OutputGate(InputGate):
    def _compute_extra_term(self, prev_states, axes, **kwargs):
        cells = kwargs['cells']
        return tf.multiply(cells, self.shared_kernel)


class CellState(ComputationGraph):
    def __init__(self, units, input_dim, ndims, kernel_initializer,
                 recurrent_initializer, activation):
        super(CellState, self).__init__(units, input_dim, ndims, kernel_initializer, recurrent_initializer)

        self.activation = activation

    def compute(self, x_batch, prev_states, axes, **kwargs):
        b_input = kwargs['b_input']
        b_forget = kwargs['b_forget']

        a = super(CellState, self).compute(x_batch, prev_states, axes)

        axis_to_forget_gate = {}
        axis_to_cell = {}
        for axis in axes:
            b, s = prev_states[axis]
            axis_to_forget_gate[axis] = b_forget[axis]
            axis_to_cell[axis] = s

        hadamard_sum = self._compute_hadamard_product_sum(axis_to_cell, axis_to_forget_gate, axes)

        return tf.add(tf.multiply(b_input, self.activation(a)), hadamard_sum)
