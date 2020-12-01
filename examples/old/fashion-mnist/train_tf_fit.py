from hub import dataset
import tensorflow as tf


def create_CNN():
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=2,
            padding="same",
            activation="relu",
            input_shape=(28, 28, 1),
        )
    )
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(
        tf.keras.layers.Conv2D(
            filters=32, kernel_size=2, padding="same", activation="relu"
        )
    )
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(10, activation="softmax"))
    return model


def to_model_fit(item):
    data = item["data"]
    data = tf.expand_dims(data, axis=2)
    labels = item["labels"]
    return (data, labels)


def main():
    BATCH_SIZE = 64
    EPOCHS = 3

    # Load data
    ds = dataset.load("mnist/fashion-mnist")

    # transform into Tensorflow dataset
    # max_text_len is an optional argument that fixes the maximum length of text labels
    ds = ds.to_tensorflow(max_text_len=15)

    # converting ds so that it can be directly used in model.fit
    ds = ds.map(lambda x: to_model_fit(x))

    # Splitting back into the original train and test sets
    train_dataset = ds.take(60000)
    test_dataset = ds.skip(60000)

    train_dataset = train_dataset.batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)

    model = create_CNN()
    # model.summary()
    model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    model.fit(
        train_dataset, epochs=EPOCHS, validation_data=test_dataset, validation_steps=1
    )


if __name__ == "__main__":
    main()
