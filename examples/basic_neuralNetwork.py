#basic Neural Network to solve a linear equation. 
#so, let inputs be x = [-1, 0, 1, 2, 3, 4]
# => Y = [-3, -1, 1, 3, 5, 7] 
#Now train a model to predict y for any x

import numpy as np
import tensorflow as tf
from tensorflow import keras

model = tf.keras.Sequential()
model.add(keras.layers.Dense(1, input_shape = [1]))

model.compile(optimizer='sgd', loss = 'mean_squared_error')

xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model.fit(xs, ys, epochs=1000)

print(model.predict([10.0]))