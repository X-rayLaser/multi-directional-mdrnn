# Quick start

The library provides 3 different multi-dimensional RNN architectures mimicking Keras API. These layers are
MDRNN, MDGRU, MDLSTM.

Create a 2-dimensional RNN (analog of simple RNN):
```
from mdrnn import MDRNN
rnn = MDRNN(units=16, input_shape=(None, None, 10), activation='tanh', return_sequences=True)
output = rnn(np.zeros((1, 5, 4, 10))
```

Create a 2-dimensional GRU (analog of GRU layer in Keras):
```
from mdrnn import MDGRU
rnn = MDGRU(units=16, input_shape=(None, None, 10), activation='tanh', return_sequences=True)
output = rnn(np.zeros((1, 5, 4, 10))
```

Create a 2-dimensional LSTM (analog of LSTM layer in Keras):
```
from mdrnn import MDLSTM
rnn = MDLSTM(units=16, input_shape=(None, None, 10), activation='tanh', return_sequences=True)
output = rnn(np.zeros((1, 5, 4, 10))
```

Create a multi-directional LSTM using MultiDirectional wrapper:
```
from mdrnn import MDLSTM, MultiDirectional
rnn = MDLSTM(units=16, input_shape=(None, None, 10), activation='tanh', return_sequences=True)
multirnn = MultiDirectional(rnn)
output = multirnn(np.zeros((1, 5, 4, 10))
```

Each of these layers can be used to build Keras sequential model and train it as usual.

```
model = tf.keras.Sequential()
model.add(MDLSTM(units=16, input_shape=(None, None, 10), activation='tanh', return_sequences=True))
model.add(TimeDistributed(Dense(units=10, activation='softmax')))
model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(...)
```