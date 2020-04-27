import tensorflow as tf
from mdrnn import MultiDirectional
from mdrnn import MDRNN
from tensorflow.keras.datasets import mnist


def down_sample(images, size):
    images = images.reshape(-1, 28, 28, 1)
    return tf.image.resize(images, [size, size]).numpy()


def fit_mdrnn(target_image_size=10, rnn_units=128, epochs=30, batch_size=32):
    # get MNIST examples
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # down sample images to speed up the training and graph building process for mdrnn
    x_train = down_sample(x_train, target_image_size)
    x_test = down_sample(x_test, target_image_size)

    inp = tf.keras.layers.Input(shape=(target_image_size, target_image_size, 1))

    # create multi-directional MDRNN layer
    rnn = MultiDirectional(MDRNN(units=rnn_units, input_shape=[target_image_size, target_image_size, 1]))

    dense = tf.keras.layers.Dense(units=10, activation='softmax')

    # build a model
    x = inp
    x = rnn(x)
    outputs = dense(x)
    model = tf.keras.Model(inp, outputs)

    # choose Adam optimizer, set gradient clipping to prevent gradient explosion,
    # set a categorical cross-entropy loss function
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001, clipnorm=100),
                  loss='categorical_crossentropy', metrics=['acc'])
    model.summary()

    # fit the model
    model.fit(x_train, tf.keras.utils.to_categorical(y_train), epochs=epochs,
              validation_data=(x_test, tf.keras.utils.to_categorical(y_test)), batch_size=batch_size)


fit_mdrnn()
