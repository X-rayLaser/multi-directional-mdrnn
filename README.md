# Multi-Directional Multi-Dimensional Recurrent Neural Networks

A library built on top of TensorFlow implementing the model described in
Alex Graves's paper https://arxiv.org/pdf/0705.2011.pdf.
The library comes with a set of custom Keras layers.
Each layer can be seamlessly used in Keras to build a model and
train it as usual.

# Status: under development

This repository is in its early stages. The code presented here is not stable yet
and it wasn't extensively tested. Use it at your own risk

# Features

Layers available now:
- **MDRNN**: layer analogous to Keras SimpleRNN layer for processing multi-dimensional inputs
- **MultiDirectional**: layer-wrapper analogous to Keras Bidirectional for creating 
multi-directional multi-dimensional RNN

Layers currently under development (coming soon):
- **MDGRU**: analogous to Keras GRU layer
- **MDLSTM**: analogous to Keras LSTM layer

Additional features:
- easy to use with Keras
- Keras-like API for each layer
- option to choose order/direction in which to process inputs
- computations are run on CPU

# Installation
Install the package from PyPI:
```
pip install mdrnn
```

Alternatively, clone the repository and install dependencies:
```
git clone <repo_url>
cd <repo_directory>
pip install -r requirements.txt
```

# Quick Start

Create a 2-dimensional RNN:
```
from mdrnn import MDRNN, MultiDirectional
import numpy as np
import tensorflow as tf
rnn = MDRNN(units=16, input_shape=(5, 4, 10), activation='tanh', return_sequences=True)
output = rnn(np.zeros((1, 5, 4, 10)))
```

Build a Keras model consisting of 1 MDRNN layer and train it:

```
model = tf.keras.Sequential()
model.add(MDRNN(units=16, input_shape=(2, 3, 6), activation='tanh'))
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))
model.compile(loss='categorical_crossentropy', metrics=['acc'])
model.summary()
x = np.zeros((10, 2, 3, 6))
y = np.zeros((10, 10,))
model.fit(x, y)
```

Similarly, create and train a multi-directional MDRNN
```
x = np.zeros((10, 2, 3, 6))
y = np.zeros((10, 40,))

model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape=(2, 3, 6)))
model.add(MultiDirectional(MDRNN(10, input_shape=[2, 3, 6])))

model.compile(loss='categorical_crossentropy', metrics=['acc'])
model.summary()

model.fit(x, y, epochs=1)
```


# Requirements

- TensorFlow version >= 2.0

# References

[1] A. Graves, S. Ferńandez, and J. Schmidhuber. Multidimensional recurrent neural networks.

[2] A. Graves and J. Schmidhuber. Offline Handwriting Recognition with Multidimensional Recurrent Neural Networks.
