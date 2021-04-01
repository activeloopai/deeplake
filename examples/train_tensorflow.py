"""Basic example of training tensorflow model on hub.Dataset
"""

import tensorflow as tf

import hub
from hub.training.model import Model


def to_model_fit(item):
    x = item["image"]
    y = item["label"]
    return (x, y)


def example_to_tensorflow():
    ds = hub.Dataset("activeloop/fashion_mnist_train")
    tds = ds.to_tensorflow(include_shapes=True).batch(8)
    tds = tds.map(lambda x: to_model_fit(x))
    return tds


def train(ds: hub.Dataset):
    net = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10),
        ]
    )
    model_cl = Model(net)
    net.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    net.fit(ds, epochs=10)
    model_cl.store("/tmp/")


ds = example_to_tensorflow()
train(ds)
