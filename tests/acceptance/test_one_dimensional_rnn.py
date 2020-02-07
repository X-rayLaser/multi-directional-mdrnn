from unittest import TestCase
import tensorflow as tf
import numpy as np
from mdrnn import MDRNN


class Rnn1DTestCase(TestCase):
    def setUp(self):
        self.x_train = np.array([
            [0, 1, 2],
            [1, 2, 3]
        ]).reshape((2, 3, 1))

        self.units = 4

        self.batch_size = 2
        self.seq_len = self.x_train.shape[1]
        self.input_dim = self.x_train.shape[-1]
        return_sequences = self.should_return_sequences()

        self.model = self.create_model(input_dim=self.input_dim, units=self.units,
                                       seq_len=self.seq_len, return_sequences=return_sequences)

    def create_model(self, input_dim, units, seq_len, return_sequences,
                     output_units=1, output_activation=None):
        model = tf.keras.Sequential()
        model.add(MDRNN(input_dim=input_dim, units=units, ndims=1,
                        input_shape=[seq_len, input_dim],
                        return_sequences=return_sequences))

        model.add(tf.keras.layers.Dense(units=output_units, activation=output_activation))
        model.compile(optimizer='sgd', loss='mean_squared_error', metrics=[])
        return model

    @property
    def expected_shape(self):
        if self.should_return_sequences():
            return self.x_train.shape

        return self.batch_size, self.input_dim

    @property
    def expected_clases_shape(self):
        return self.expected_shape

    @property
    def y_train(self):
        if self.should_return_sequences():
            return np.array([[3, 3, 3], [6, 6, 6]]).reshape((-1, 3, 1))

        return np.array([3, 6]).reshape(-1, 1)

    def should_return_sequences(self):
        return False

    def make_generator(self):
        while True:
            for i in range(len(self.x_train)):
                x = self.x_train[i].reshape(1, self.seq_len, self.input_dim)
                y = self.y_train[i].reshape(1, -1)
                yield x, y


class RnnWithoutReturningSequencesTests(Rnn1DTestCase):
    def test_predict_probabilities(self):
        last_output = self.model.predict(self.x_train)
        self.assertEqual(self.expected_shape, last_output.shape)

    def test_predict_classes(self):
        if hasattr(self.model, 'predict_classes'):
            last_output = self.model.predict_classes(self.x_train)
            self.assertEqual(self.expected_clases_shape, last_output.shape)

    def test_predict_generator(self):
        res = self.model.predict_generator(self.make_generator(),
                                           steps=self.batch_size)
        self.assertEqual(self.expected_shape, res.shape)

    def test_predict_on_batch(self):
        res = self.model.predict_on_batch(self.x_train)
        self.assertEqual(self.expected_shape, res.shape)

    def test_predict_proba(self):
        if hasattr(self.model, 'predict_proba'):
            res = self.model.predict_proba(self.x_train)
            self.assertEqual(self.expected_shape, res.shape)

    def test_fit(self):
        self.model.fit(self.x_train, self.y_train, epochs=100)

    def test_fit_method_using_generator(self):
        self.model.fit(self.make_generator(),
                       steps_per_epoch=self.batch_size, epochs=100)

    def test_fit_generator_method(self):
        self.model.fit_generator(self.make_generator(),
                                 steps_per_epoch=self.batch_size, epochs=100)

    def test_train_on_batch(self):
        self.model.train_on_batch(self.x_train, self.y_train)

    def test_test_on_batch_method(self):
        loss = self.model.test_on_batch(self.x_train, self.y_train)
        self.assertIsNotNone(loss)

    def test_evaluate_method(self):
        loss = self.model.evaluate(self.x_train, self.y_train)
        self.assertIsNotNone(loss)

    def test_evaluate_method_using_generator(self):
        loss = self.model.evaluate(self.make_generator(), steps=self.batch_size)
        self.assertIsNotNone(loss)

    def test_evaluate_generator_method(self):
        loss = self.model.evaluate_generator(self.make_generator(), steps=self.batch_size)
        self.assertIsNotNone(loss)


class RNNSequenceTests(RnnWithoutReturningSequencesTests):
    def should_return_sequences(self):
        return True


class FunctionalRNNTests(RnnWithoutReturningSequencesTests):
    def create_model(self, input_dim, units, seq_len, return_sequences,
                     output_units=1, output_activation=None):

        inp = tf.keras.layers.Input(shape=[seq_len, input_dim])

        x = inp

        mdrnn = MDRNN(input_dim=input_dim, units=units, ndims=1,
                      return_sequences=return_sequences)

        densor = tf.keras.layers.Dense(units=output_units, activation=output_activation)

        x = mdrnn(x)

        output = densor(x)

        model = tf.keras.Model(inputs=inp, outputs=output)
        model.compile(optimizer='sgd', loss='mean_squared_error', metrics=[])
        return model
