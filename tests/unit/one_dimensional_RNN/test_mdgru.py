from unittest import TestCase
from tensorflow.keras import initializers
import tensorflow as tf
import numpy as np
from mdrnn import MultiDirectional, Direction
from mdrnn._layers.gru import MDGRU


def make_rnns(return_sequences, return_state, go_backwards=False):
        seed = 1
        kernel_initializer = initializers.glorot_uniform(seed)
        recurrent_initializer = initializers.he_normal(seed)
        bias_initializer = initializers.he_normal(seed)

        kwargs = dict(units=3, input_shape=(None, 5),
                      activation='tanh',
                      recurrent_activation='sigmoid',
                      kernel_initializer=kernel_initializer,
                      recurrent_initializer=recurrent_initializer,
                      bias_initializer=bias_initializer,
                      return_sequences=return_sequences,
                      return_state=return_state
                      )

        if go_backwards:
            kwargs.update(dict(direction=Direction(-1)))

        rnn = MDGRU(**kwargs)
        keras_rnn = tf.keras.layers.GRU(units=3, activation='tanh',
                                        recurrent_activation='sigmoid', implementation=1,
                                        kernel_initializer=kernel_initializer,
                                        recurrent_initializer=recurrent_initializer,
                                        bias_initializer=bias_initializer,
                                        return_sequences=return_sequences,
                                        reset_after=False,
                                        return_state=return_state,
                                        go_backwards=go_backwards)
        return rnn, keras_rnn


class SimpleTestTemplate(object):
    def __init__(self, return_sequences=False):
        self.return_sequences = return_sequences
        self.return_state = False

    def run(self, x, initial_state=None):
        rnn, keras_rnn = self.make_rnns()

        rnn_result = rnn(x, initial_state)
        rnn_result = self.post_process_mdrnn_result(rnn_result)

        keras_rnn_result = keras_rnn(x, initial_state)
        keras_rnn_result = self.post_process_keras_result(keras_rnn_result)

        self.assert_outputs_equal(keras_rnn_result, rnn_result)

    def make_rnns(self):
        return make_rnns(return_sequences=self.return_sequences,
                         return_state=self.return_state, go_backwards=False)

    def post_process_mdrnn_result(self, result):
        return result.numpy()

    def post_process_keras_result(self, result):
        return result.numpy()

    def assert_outputs_equal(self, expected, actual):
        np.testing.assert_almost_equal(expected, actual, 6)


class ReturnStateTemplate(SimpleTestTemplate):
    def __init__(self, return_sequences=False):
        super(ReturnStateTemplate, self).__init__(return_sequences)
        self.return_state = True

    def post_process_mdrnn_result(self, result):
        output, state = result
        return output.numpy(), state.numpy()

    def post_process_keras_result(self, result):
        return self.post_process_mdrnn_result(result)

    def assert_outputs_equal(self, expected, actual):
        assert len(expected) == len(actual)
        assert len(expected) == 2
        np.testing.assert_almost_equal(expected[0], actual[0], 6)
        np.testing.assert_almost_equal(expected[1], actual[1], 6)


class GoBackwardsTemplate(SimpleTestTemplate):
    def make_rnns(self):
        return make_rnns(return_sequences=self.return_sequences,
                         return_state=self.return_state, go_backwards=True)

    def post_process_mdrnn_result(self, result):
        array = result.numpy()
        if self.return_sequences:
            array = reverse_second_axis(array)
        return array


class GoBackwardsAndReturnStateTemplate(ReturnStateTemplate):
    def make_rnns(self):
        return make_rnns(return_sequences=self.return_sequences,
                         return_state=self.return_state, go_backwards=True)

    def post_process_mdrnn_result(self, result):
        output, state = super(GoBackwardsAndReturnStateTemplate, self).post_process_mdrnn_result(result)

        if self.return_sequences:
            output = reverse_second_axis(output)
        return output, state

    def post_process_keras_result(self, result):
        output, state = result
        return output.numpy(), state.numpy()


def reverse_second_axis(a):
    size = a.shape[1]
    indices = np.arange(size - 1, -1, -1)
    return a[:, indices, :]


class UnidirectionalOneDimensionalGRUTests(TestCase):
    def test_outputs_on_single_step_sequence(self):
        x = tf.constant(np.random.rand(3, 1, 5), dtype=tf.float32)
        test_template = SimpleTestTemplate()
        test_template.run(x)

    def test_outputs_on_many_steps_sequence(self):
        x = tf.constant(np.random.rand(3, 4, 5), dtype=tf.float32)
        test_template = SimpleTestTemplate()
        test_template.run(x)

    def test_processing_sequence_backwards(self):
        x = tf.constant(np.random.rand(3, 4, 5), dtype=tf.float32)
        test_template = GoBackwardsTemplate(return_sequences=False)
        test_template.run(x)

    def test_with_flags_return_sequences_and_go_backwards(self):
        x = tf.constant(np.random.rand(3, 4, 5), dtype=tf.float32)
        test_template = GoBackwardsTemplate(return_sequences=True)
        test_template.run(x)

    def test_with_flag_return_sequences(self):
        x = tf.constant(np.random.rand(3, 4, 5), dtype=tf.float32)
        test_template = SimpleTestTemplate(return_sequences=True)
        test_template.run(x)

    def test_with_flag_return_state(self):
        x = tf.constant(np.random.rand(3, 4, 5), dtype=tf.float32)
        test_template = ReturnStateTemplate()
        test_template.run(x)

    def test_with_both_flags_return_sequences_and_return_state(self):
        x = tf.constant(np.random.rand(3, 4, 5), dtype=tf.float32)
        test_template = ReturnStateTemplate(return_sequences=True)
        test_template.run(x)

    def test_with_flags_return_sequences_and_return_state_and_go_backwards(self):
        x = tf.constant(np.random.rand(3, 4, 5), dtype=tf.float32)
        test_template = GoBackwardsAndReturnStateTemplate(return_sequences=True)
        test_template.run(x)


class UnidirectionalInitialStateTests(TestCase):
    def setUp(self):
        self.batch_size = 3
        self.x = tf.constant(np.random.rand(self.batch_size, 4, 5), dtype=tf.float32)
        self.initial_state = self.make_random_initial_state()

    def make_random_initial_state(self):
        a = np.random.rand(self.batch_size, 3)
        return tf.constant(a, dtype=tf.float32)

    def test_outputs_on_single_step_sequence(self):
        x = tf.constant(np.random.rand(3, 1, 5), dtype=tf.float32)
        test_template = SimpleTestTemplate(return_sequences=False)
        test_template.run(x, initial_state=self.make_random_initial_state())

    def test_outputs_on_many_steps_sequence(self):
        test_template = SimpleTestTemplate(return_sequences=False)
        test_template.run(self.x, self.initial_state)

    def test_with_flag_return_sequences(self):
        test_template = SimpleTestTemplate(return_sequences=True)
        test_template.run(self.x, self.initial_state)

    def test_with_flag_return_state(self):
        test_template = ReturnStateTemplate(return_sequences=False)
        test_template.run(self.x, self.initial_state)

    def test_with_both_flags_return_sequences_and_return_state(self):
        test_template = ReturnStateTemplate(return_sequences=True)
        test_template.run(self.x, self.initial_state)


class InputTypeTests(TestCase):
    def assert_correct_behavior(self, x):
        test_template = SimpleTestTemplate(return_sequences=False)
        test_template.run(x)

        rnn, keras_rnn = make_rnns(return_sequences=False, return_state=False)
        rnn.call(x)

    def test_can_use_numpy_array_as_input(self):
        x = np.random.rand(3, 4, 5)
        self.assert_correct_behavior(x)

    def test_can_use_float32_tensor_as_input(self):
        x = np.random.rand(3, 4, 5)
        x = tf.constant(x, tf.float32)
        self.assert_correct_behavior(x)

    def test_can_use_float64_tensor_as_input(self):
        x = np.random.rand(3, 4, 5)
        x = tf.constant(x, tf.float64)
        self.assert_correct_behavior(x)


class BidirectionalMDGRUTests(TestCase):
    def make_rnns(self, return_sequences, return_state, go_backwards=False):
        rnn, keras_rnn = make_rnns(return_sequences, return_state, go_backwards)

        rnn = MultiDirectional(rnn)
        keras_rnn = tf.keras.layers.Bidirectional(keras_rnn, merge_mode='concat')
        return rnn, keras_rnn

    def assert_rnn_outputs_equal(self, x, initial_state=None, **rnn_kwargs):
        rnn, keras_rnn = self.make_rnns(**rnn_kwargs)

        rnn_result = rnn(x, initial_state=initial_state)
        keras_rnn_result = keras_rnn(x, initial_state=initial_state)

        self.assertEqual(len(rnn_result), len(keras_rnn_result))
        np.testing.assert_almost_equal(rnn_result.numpy(), keras_rnn_result.numpy(), 6)

    def assert_output_tuples_equal(self, x, initial_state=None, **rnn_kwargs):
        rnn, keras_rnn = self.make_rnns(**rnn_kwargs)

        rnn_result = rnn(x, initial_state=initial_state)
        keras_rnn_result = keras_rnn(x, initial_state=initial_state)

        self.assertEqual(len(rnn_result), 3)
        self.assertEqual(len(keras_rnn_result), 3)

        np.testing.assert_almost_equal(rnn_result[0].numpy(), keras_rnn_result[0].numpy(), 6)
        np.testing.assert_almost_equal(rnn_result[1].numpy(), keras_rnn_result[1].numpy(), 6)
        np.testing.assert_almost_equal(rnn_result[2].numpy(), keras_rnn_result[2].numpy(), 6)

    def test_outputs_on_many_steps_sequence(self):
        x = tf.constant(np.random.rand(3, 4, 5), dtype=tf.float32)
        self.assert_rnn_outputs_equal(x, return_sequences=False, return_state=False)

    def test_with_flag_return_sequences(self):
        x = tf.constant(np.random.rand(3, 4, 5), dtype=tf.float32)
        self.assert_rnn_outputs_equal(x, return_sequences=True, return_state=False)

    def test_with_flag_return_state(self):
        x = tf.constant(np.random.rand(3, 4, 5), dtype=tf.float32)
        self.assert_output_tuples_equal(x, return_sequences=False, return_state=True)

    def test_with_both_flags_return_sequences_and_return_state(self):
        x = tf.constant(np.random.rand(3, 4, 5), dtype=tf.float32)
        self.assert_output_tuples_equal(x, return_sequences=True, return_state=True)
