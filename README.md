# Multi-Directional Multi-Dimensional Recurrent Neural Networks

A library built on top of TensorFlow implementing the model described in
Alex Graves's paper https://arxiv.org/pdf/0705.2011.pdf.
Specifically, it implements M-dimensional recurrent neural
networks that can process inputs with any number of dimensions (2D, 3D, 6D, etc.).
The library comes with a set of custom Keras layers.
Each layer can be seamlessly used in Keras to build a model and
train it as usual.

It requires a TensorFlow version 2.0 or greater and it was tested with Python>=3.6.8.

# Status: under development

This repository is in its early stages. The code presented here is not stable yet
and it wasn't extensively tested. Use it at your own risk

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
from mdrnn import MDRNN
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

# Features

Layers available now:
- **MDRNN**:  layer analogous to Keras SimpleRNN layer for processing multi-dimensional inputs

Layers currently under development (coming soon):
- **MDGRU**: analogous to Keras GRU layer
- **MDLSTM**: analogous to Keras LSTM layer
- **MultiDirectional**: layer-wrapper creating multi-directional multi-dimensional RNN/GRU/LSTM

Additional features:
- Keras-like API for each layer
- choosing order/direction in which input should be processed
- maximally general implementation: it can process input with any number of dimensions
as long as there is enough memory

# References:

[1]. A. Graves, S. Ferńandez, and J. Schmidhuber. Multidimensional recurrent neural networks.

[2]. A. Graves and J. Schmidhuber. Offline Handwriting Recognition with Multidimensional Recurrent Neural Networks.
