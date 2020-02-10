from unittest import TestCase
import numpy as np
from mdrnn import MDRNN
from tensorflow.keras import initializers


class Degenerate2DInput(TestCase):
    def test_on_1x1_input(self):
        seed = 1

        kwargs = dict(units=3, input_shape=(None, None, 2),
                      kernel_initializer=initializers.glorot_uniform(seed),
                      recurrent_initializer=initializers.he_normal(seed),
                      bias_initializer=initializers.Constant(2),
                      return_sequences=True,
                      activation='tanh'
                      )

        x = np.array([5, 10], dtype=np.float)
        x_1d = x.reshape((1, 1, 2))
        x_2d = x.reshape((1, 1, 1, 2))

        rnn2d = MDRNN(**kwargs)

        keras_kwargs = dict(kwargs)
        keras_kwargs.update(dict(input_shape=(None, 2)))
        keras_rnn = MDRNN(**keras_kwargs)

        actual = rnn2d(x_2d).numpy()
        desired = keras_rnn(x_1d).numpy().reshape((1, 1, 1, 3))
        np.testing.assert_almost_equal(actual, desired, 6)
