from unittest import TestCase
import tensorflow as tf
import numpy as np
from mdrnn import MDRNN


class RnnLastOutputPredictionTests(TestCase):
    def setUp(self):
        self.x = np.zeros((5, 4, 3))
        model = tf.keras.Sequential()
        model.add(MDRNN(input_dim=3, units=2, ndims=1, input_shape=[4, 3]))
        self.model = model
        self.expected_shape = (5, 2)
        self.expected_clases_shape = (5, )

    def test_build_model_and_predict_probabilities(self):
        last_output = self.model.predict(self.x)
        self.assertEqual(self.expected_shape, last_output.shape)

    def test_build_model_and_predict_classes(self):
        last_output = self.model.predict_classes(self.x)
        self.assertEqual(self.expected_clases_shape, last_output.shape)

    def test_predict_generator(self):
        def make_gen():
            for inp in self.x:
                yield inp.reshape(1, 4, 3), np.zeros((2, 2))

        res = self.model.predict_generator(make_gen(), steps=5)
        self.assertEqual(self.expected_shape, res.shape)

    def test_predict_on_batch(self):
        res = self.model.predict_on_batch(self.x)
        self.assertEqual(self.expected_shape, res.shape)

    def test_predict_proba(self):
        res = self.model.predict_proba(self.x)
        self.assertEqual(self.expected_shape, res.shape)


class RNNSequencePredictionTests(RnnLastOutputPredictionTests):
    def setUp(self):
        self.x = np.zeros((5, 4, 3))
        model = tf.keras.Sequential()
        model.add(MDRNN(input_dim=3, units=2, ndims=1, input_shape=[4, 3],
                        return_sequences=True))
        self.model = model
        self.expected_shape = (5, 4, 2)
        self.expected_clases_shape = (5, 4)


class OneDimensionalRnnFittingTests(TestCase):
    def setUp(self):
        self.x_train = np.array([
            [0, 1, 2],
            [1, 2, 3]
        ]).reshape((2, 3, 1))

        self.y_train = np.array([3, 6]).reshape(-1, 1)

        model = tf.keras.Sequential()
        model.add(MDRNN(input_dim=1, units=8, ndims=1, input_shape=[3, 1]))
        model.add(tf.keras.layers.Dense(units=1, activation=None))
        model.compile(optimizer='sgd', loss='mean_squared_error', metrics=[])
        self.model = model

    def make_generator(self):
        while True:
            for i in range(2):
                x = self.x_train[i].reshape(1, 3, 1)
                y = self.y_train[i].reshape(1, -1)
                yield x, y

    def test_fit(self):
        self.model.fit(self.x_train, self.y_train, epochs=100)

    def test_fit_method_using_generator(self):
        self.model.fit(self.make_generator(), steps_per_epoch=2, epochs=100)

    def test_fit_generator_method(self):
        self.model.fit_generator(self.make_generator(), steps_per_epoch=2, epochs=100)

    def test_train_on_batch(self):
        self.model.train_on_batch(self.x_train, self.y_train)


class RNNEvaluationTests(TestCase):
    def setUp(self):
        self.x_train = np.array([
            [0, 1, 2],
            [1, 2, 3]
        ]).reshape((2, 3, 1))

        self.y_train = np.array([3, 6]).reshape(-1, 1)

        model = tf.keras.Sequential()
        model.add(MDRNN(input_dim=1, units=8, ndims=1, input_shape=[3, 1]))
        model.add(tf.keras.layers.Dense(units=1, activation=None))
        model.compile(optimizer='sgd', loss='mean_squared_error', metrics=[])
        self.model = model

    def make_generator(self):
        while True:
            for i in range(2):
                x = self.x_train[i].reshape(1, 3, 1)
                y = self.y_train[i].reshape(1, -1)
                yield x, y

    def test_test_on_batch_method(self):
        loss = self.model.test_on_batch(self.x_train, self.y_train)
        self.assertIsNotNone(loss, float)

    def test_evaluate_method(self):
        loss = self.model.evaluate(self.x_train, self.y_train)
        self.assertIsNotNone(loss, float)

    def test_evaluate_method_using_generator(self):
        loss = self.model.evaluate(self.make_generator(), steps=2)
        self.assertIsNotNone(loss, float)

    def test_evaluate_generator_method(self):
        loss = self.model.evaluate_generator(self.make_generator(), steps=2)
        self.assertIsNotNone(loss, float)
