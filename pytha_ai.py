import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import numpy as np
import matplotlib.pyplot as plt

inputs = Input(shape=(2,))
hidden = Dense(64, activation='relu')(inputs)
outputs = Dense(1, activation='relu')(hidden)

model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam', loss='mean_squared_error')

data = np.random.randint(1000, size=(10000000, 2), dtype='int32')
labels = []
for x in data:
    true_value = (x[0]**2+x[1]**2)**0.5
    labels.append(true_value)

labels = np.array(labels, dtype='float32')

model.fit(data, labels, epochs=10, batch_size=100000)

model.save('my_model.h5')
