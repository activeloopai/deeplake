from datetime import datetime
from packaging import version

import os

import tensorflow as tf

print("TensorFlow version: ", tf.__version__)

device_name = tf.test.gpu_device_name()
# if not device_name:
 #  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

import tensorflow_datasets as tfds
tfds.disable_progress_bar()

(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)
import hub

ds_train = hub.load("hub://activeloop/mnist-train").tensorflow()
ds_test = hub.load("hub://activeloop/mnist-train").tensorflow()

def normalize_img(sample):
  images, labels = sample['images'], sample['labels']
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(images, tf.float32) / 255., labels

ds_train = ds_train.map(normalize_img)
ds_train = ds_train.batch(128)

ds_test = ds_test.map(normalize_img)
ds_test = ds_test.batch(128)

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
  tf.keras.layers.Dense(128,activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=['accuracy']
)

# Create a TensorBoard callback
logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
                                                 histogram_freq = 1,
                                                 profile_batch = '500,520')

model.fit(ds_train,
          epochs=2,
          validation_data=ds_test,
          callbacks = [tboard_callback])