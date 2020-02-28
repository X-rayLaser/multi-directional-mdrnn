import tensorflow as tf

from mdrnn._util.directions import Direction


class MultiDirectional(tf.keras.layers.Layer):
    def __init__(self, rnn, **kwargs):
        super(MultiDirectional, self).__init__(**kwargs)

        self._original_rnn = rnn
        directions = Direction.get_all_directions(rnn.ndims)
        self._rnns = [rnn.spawn(direction) for direction in directions]

    def build(self, input_shape):
        pass

    def call(self, inputs, **kwargs):
        if self._original_rnn.return_sequences:
            num_output_dimensions = len(inputs.shape)
        else:
            num_output_dimensions = 2

        results_list = [rnn.call(inputs, **kwargs) for rnn in self._rnns]

        if not self._original_rnn.return_state:
            last_axis = num_output_dimensions - 1
            return tf.concat(results_list, axis=last_axis)
        else:
            outputs_list = []
            states_list = []

            for result in results_list:
                activations = result[0]
                states = result[1]
                outputs_list.append(activations)
                states_list.append(states)

            outputs_last_axis = num_output_dimensions - 1
            outputs = tf.concat(outputs_list, axis=outputs_last_axis)

            return [outputs] + states_list