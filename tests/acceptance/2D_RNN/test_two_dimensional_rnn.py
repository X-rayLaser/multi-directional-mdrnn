from unittest import TestCase
from mdrnn import MDRNN, MultiDirectional
import numpy as np
import tensorflow as tf


class FeedForwardTests(TestCase):
    def test_feed_uni_directional(self):
        rnn = MDRNN(units=16, input_shape=(5, 4, 10), activation='tanh', return_sequences=True)
        output = rnn(np.zeros((1, 5, 4, 10)))
        self.assertEqual((1, 5, 4, 16), output.shape)

    def test_feed_multi_directional_rnn(self):
        rnn = MultiDirectional(MDRNN(units=16, input_shape=(5, 4, 10), activation='tanh', return_sequences=True))
        output = rnn(np.zeros((1, 5, 4, 10)))
        self.assertEqual((1, 5, 4, 16 * 4), output.shape)


class FittingTests(TestCase):
    def test_fit_uni_directional(self):
        model = tf.keras.Sequential()
        model.add(MDRNN(units=16, input_shape=(2, 3, 6), activation='tanh'))
        model.add(tf.keras.layers.Dense(units=10, activation='softmax'))
        model.compile(loss='categorical_crossentropy', metrics=['acc'])
        model.summary()
        x = np.zeros((10, 2, 3, 6))
        y = np.zeros((10, 10,))
        model.fit(x, y)

    def test_fit_multi_directional(self):
        x = np.zeros((10, 2, 3, 6))
        y = np.zeros((10, 40,))

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(2, 3, 6)))
        model.add(MultiDirectional(MDRNN(10, input_shape=[2, 3, 6])))

        model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001, clipnorm=100), loss='categorical_crossentropy',
                      metrics=['acc'])
        model.summary()

        model.fit(x, y, epochs=1)
