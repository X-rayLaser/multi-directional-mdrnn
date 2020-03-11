from unittest import TestCase
from tensorflow.keras import initializers
import tensorflow as tf
import numpy as np
from mdrnn import MDGRU, MultiDirectional, Direction


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


class UnidirectionalOneDimensionalGRUTests(TestCase):
    def make_rnns(self, return_sequences, return_state, go_backwards=False):
        return make_rnns(return_sequences, return_state, go_backwards)

    def assert_rnn_outputs_equal(self, x, **rnn_kwargs):
        rnn, keras_rnn = self.make_rnns(**rnn_kwargs)

        rnn_result = rnn(x)
        keras_rnn_result = keras_rnn(x)

        self.assertEqual(len(rnn_result), len(keras_rnn_result))
        np.testing.assert_almost_equal(rnn_result.numpy(), keras_rnn_result.numpy(), 6)

    def assert_output_tuples_equal(self, x, **rnn_kwargs):
        rnn, keras_rnn = self.make_rnns(**rnn_kwargs)

        rnn_result = rnn(x)
        keras_rnn_result = keras_rnn(x)

        self.assertEqual(len(rnn_result), 2)
        self.assertEqual(len(keras_rnn_result), 2)

        np.testing.assert_almost_equal(rnn_result[0].numpy(), keras_rnn_result[0].numpy(), 6)
        np.testing.assert_almost_equal(rnn_result[1].numpy(), keras_rnn_result[1].numpy(), 6)

    def test_outputs_on_single_step_sequence(self):
        x = tf.constant(np.random.rand(3, 1, 5), dtype=tf.float32)
        self.assert_rnn_outputs_equal(x, return_sequences=False, return_state=False)

    def test_outputs_on_many_steps_sequence(self):
        x = tf.constant(np.random.rand(3, 4, 5), dtype=tf.float32)
        self.assert_rnn_outputs_equal(x, return_sequences=False, return_state=False)

    def test_processing_sequence_backwards(self):
        x = tf.constant(np.random.rand(3, 4, 5), dtype=tf.float32)
        self.assert_rnn_outputs_equal(x, return_sequences=False, return_state=False, go_backwards=True)

    def test_with_flags_return_sequences_and_go_backwards(self):
        x = tf.constant(np.random.rand(3, 4, 5), dtype=tf.float32)
        self.assert_rnn_outputs_equal(x, return_sequences=True, return_state=False, go_backwards=True)

    def test_with_flag_return_sequences(self):
        x = tf.constant(np.random.rand(3, 4, 5), dtype=tf.float32)
        self.assert_rnn_outputs_equal(x, return_sequences=True, return_state=False)

    def test_with_flag_return_state(self):
        x = tf.constant(np.random.rand(3, 4, 5), dtype=tf.float32)
        self.assert_output_tuples_equal(x, return_sequences=False, return_state=True)

    def test_with_both_flags_return_sequences_and_return_state(self):
        x = tf.constant(np.random.rand(3, 4, 5), dtype=tf.float32)
        self.assert_output_tuples_equal(x, return_sequences=True, return_state=True)

    def test_with_flags_return_sequences_and_return_state_and_go_backwards(self):
        x = tf.constant(np.random.rand(3, 4, 5), dtype=tf.float32)
        self.assert_output_tuples_equal(x, return_sequences=True, return_state=True, go_backwards=True)


class BidirectionalMDGRUTests(TestCase):
    def make_rnns(self, return_sequences, return_state, go_backwards=False):
        rnn, keras_rnn = make_rnns(return_sequences, return_state, go_backwards)

        rnn = MultiDirectional(rnn)
        keras_rnn = tf.keras.layers.Bidirectional(keras_rnn, merge_mode='concat')
        return rnn, keras_rnn

    def assert_rnn_outputs_equal(self, x, **rnn_kwargs):
        rnn, keras_rnn = self.make_rnns(**rnn_kwargs)

        rnn_result = rnn(x)
        keras_rnn_result = keras_rnn(x)

        self.assertEqual(len(rnn_result), len(keras_rnn_result))
        np.testing.assert_almost_equal(rnn_result.numpy(), keras_rnn_result.numpy(), 6)

    def assert_output_tuples_equal(self, x, **rnn_kwargs):
        rnn, keras_rnn = self.make_rnns(**rnn_kwargs)

        rnn_result = rnn(x)
        keras_rnn_result = keras_rnn(x)

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
