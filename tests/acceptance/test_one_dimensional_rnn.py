from unittest import TestCase
import tensorflow as tf
import numpy as np
from mdrnn import MDRNN


class OneDimensionalRnnTests(TestCase):
    def test_build_model_and_predict_probabilities(self):
        model = tf.keras.Sequential()
        model.add(MDRNN(input_dim=3, units=2, ndims=1))

        x = np.zeros((1, 4, 3))
        last_output = model.predict(x)

        self.assertEqual((1, 2), last_output.shape)
