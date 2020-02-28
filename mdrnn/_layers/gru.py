from tensorflow.keras.layers import Layer
import tensorflow as tf


class MDGRU(Layer):
    def __init__(self, units, input_shape, kernel_initializer=None,
                 recurrent_initializer=None, bias_initializer=None, activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 return_sequences=False, return_state=False, **kwargs):
        super().__init__(**kwargs)
        self.units = units

        input_size = input_shape[-1]
        self.Wz = tf.Variable(kernel_initializer((input_size, units)), dtype=tf.float32)
        self.Wr = tf.Variable(kernel_initializer((input_size, units)), dtype=tf.float32)
        self.Wh = tf.Variable(kernel_initializer((input_size, units)), dtype=tf.float32)

        self.Uz = tf.Variable(recurrent_initializer((units, units)), dtype=tf.float32)
        self.Ur = tf.Variable(recurrent_initializer((units, units)), dtype=tf.float32)
        self.Uh = tf.Variable(recurrent_initializer((units, units)), dtype=tf.float32)

        self.Bz = tf.Variable(bias_initializer((1, units)), dtype=tf.float32)
        self.Br = tf.Variable(bias_initializer((1, units)), dtype=tf.float32)
        self.Bh = tf.Variable(bias_initializer((1, units)), dtype=tf.float32)

        self.activation = tf.keras.activations.get(activation)
        self.recurrent_activation = tf.keras.activations.get(recurrent_activation)

    def call(self, inputs, initial_state=None, **kwargs):
        a = tf.zeros((1, self.units))
        x = tf.constant(inputs, dtype=tf.float32)

        x = tf.reshape(x, shape=[-1, x.shape[-1]])

        term = tf.add(tf.matmul(x, self.Wz), self.Bz)
        z = self.recurrent_activation(tf.add(tf.matmul(a, self.Uz), term))

        term = tf.add(tf.matmul(x, self.Wr), self.Br)
        r = self.recurrent_activation(tf.add(tf.matmul(a, self.Ur), term))

        term = tf.add(tf.matmul(x, self.Wh), self.Bh)
        h = self.activation(tf.add(tf.matmul(tf.multiply(r, a), self.Uh), term))

        forget_gate = tf.subtract(1, z)

        return tf.add(tf.multiply(z, a), tf.multiply(forget_gate, h))
