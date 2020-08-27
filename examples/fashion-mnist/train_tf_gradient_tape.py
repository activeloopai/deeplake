from hub import dataset
import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam


def create_CNN():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    return model


def train(model, train_dataset, optimizer, loss_fn, train_acc_metric):
    for batch in train_dataset:
        with tf.GradientTape() as tape:
            pred = model(tf.expand_dims(batch["data"], axis=3))
            loss = loss_fn(batch["labels"], pred)

        # calculate gradients and update the model weights
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_acc_metric.update_state(batch["labels"], pred)

    train_acc = train_acc_metric.result()
    print("Training acc: %.4f" % (float(train_acc),))
    train_acc_metric.reset_states()


def test(model, test_dataset, test_acc_metric):
    print("Evaluating on Test Set")
    for batch in test_dataset:
        pred = model(tf.expand_dims(batch["data"], axis=3), training=False)
        test_acc_metric.update_state(batch["labels"], pred)

    test_acc = test_acc_metric.result()
    print("Test acc: %.4f" % (float(test_acc),))
    test_acc_metric.reset_states()


def main():
    BATCH_SIZE = 64
    EPOCHS = 3

    optimizer = Adam()
    train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    test_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    loss_fn = SparseCategoricalCrossentropy()

    # Load data
    ds = dataset.load("abhinavtuli/fashion-mnist")

    # transform into Tensorflow dataset
    ds = ds.to_tensorflow()

    # Splitting back into the original train and test sets
    train_dataset = ds.take(60000)
    test_dataset = ds.skip(60000)

    train_dataset = train_dataset.batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)

    model = create_CNN()
    # model.summary()

    for epoch in range(EPOCHS):
        print("\nStarting Training Epoch {}".format(epoch))
        train(model, train_dataset, optimizer, loss_fn, train_acc_metric)
        print("Training Epoch {} finished\n".format(epoch))
        test(model, test_dataset, test_acc_metric)


if __name__ == "__main__":
    main()
