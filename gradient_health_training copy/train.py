import json
import shutil
import os
import pickle
from callback import MultipleClassAUROC, MultiGPUModelCheckpoint
from configparser import ConfigParser
from generator import AugmentedImageSequence
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.optimizers import Adam

# from keras.utils import multi_gpu_model
from models.keras import ModelFactory
from utility import get_sample_counts
from weights import get_class_weights
from augmenter import augmenter
import tensorflow as tf
import hub
import numpy as np

import wandb
wandb.init()


def train_gen():
    for _ in range(10000000):
        yield np.ones((512, 512, 3)), np.ones((14,))


def val_gen():
    for _ in range(0):
        yield np.ones((512, 512, 3)), np.ones((14,))


dummy_train = tf.data.Dataset.from_generator(
    train_gen,
    output_signature=(
        tf.TensorSpec(shape=(512, 512, 3), dtype=tf.uint16),
        tf.TensorSpec(shape=(14,), dtype=tf.int32),
    ))

dummy_val = tf.data.Dataset.from_generator(
    val_gen,
    output_signature=(
        tf.TensorSpec(shape=(512, 512, 3), dtype=tf.uint16),
        tf.TensorSpec(shape=(14,), dtype=tf.int32),
    ))

dummy_train = dummy_train.batch(8)
dummy_val = dummy_val.batch(8)


def only_frontal(sample):
    viewPosition = sample["viewPosition"].compute(True)
    return True if "PA" in viewPosition or "AP" in viewPosition else False


def get_image(viewPosition, images):
    for i, vp in enumerate(viewPosition):
        if vp in [5, 12]:
            return np.concatenate((images[i], images[i], images[i]), axis=2)


def to_model_fit(sample):
    viewPosition = sample["viewPosition"]
    images = sample["image"]
    image = tf.py_function(get_image, [viewPosition, images], tf.uint16)
    labels = sample["label_chexpert"]
    return image, labels


def main():
    # parser config
    config_file = "./config.ini"
    cp = ConfigParser()
    cp.read(config_file)

    # default config
    output_dir = cp["DEFAULT"].get("output_dir")
    image_source_dir = cp["DEFAULT"].get("image_source_dir")
    base_model_name = cp["DEFAULT"].get("base_model_name")
    print(base_model_name)
    class_names = cp["DEFAULT"].get("class_names").split(",")

    # train config
    use_base_model_weights = cp["TRAIN"].getboolean("use_base_model_weights")
    use_trained_model_weights = cp["TRAIN"].getboolean("use_trained_model_weights")
    use_best_weights = cp["TRAIN"].getboolean("use_best_weights")
    output_weights_name = cp["TRAIN"].get("output_weights_name")
    epochs = cp["TRAIN"].getint("epochs")
    batch_size = cp["TRAIN"].getint("batch_size")
    initial_learning_rate = cp["TRAIN"].getfloat("initial_learning_rate")
    generator_workers = cp["TRAIN"].getint("generator_workers")
    image_dimension = cp["TRAIN"].getint("image_dimension")
    train_steps = cp["TRAIN"].get("train_steps")
    patience_reduce_lr = cp["TRAIN"].getint("patience_reduce_lr")
    min_lr = cp["TRAIN"].getfloat("min_lr")
    validation_steps = cp["TRAIN"].get("validation_steps")
    positive_weights_multiply = cp["TRAIN"].getfloat("positive_weights_multiply")
    dataset_csv_dir = cp["TRAIN"].get("dataset_csv_dir")
    # if previously trained weights is used, never re-split
    if use_trained_model_weights:
        # resuming mode
        print("** use trained model weights **")
        # load training status for resuming
        training_stats_file = os.path.join(output_dir, ".training_stats.json")
        if os.path.isfile(training_stats_file):
            # TODO: add loading previous learning rate?
            training_stats = json.load(open(training_stats_file))
        else:
            training_stats = {}
    else:
        # start over
        training_stats = {}

    show_model_summary = cp["TRAIN"].getboolean("show_model_summary")
    # end parser config

    # check output_dir, create it if not exists
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    running_flag_file = os.path.join(output_dir, ".training.lock")
    if os.path.isfile(running_flag_file):
        raise RuntimeError("A process is running in this directory!!!")
    else:
        open(running_flag_file, "a").close()

    try:
        print(f"backup config file to {output_dir}")
        shutil.copy(
            config_file, os.path.join(output_dir, os.path.split(config_file)[1])
        )

        ds = hub.Dataset(
            "s3://snark-gradient-raw-data/output_single_8_5000_samples_max_4_boolean_m5_fixed/ds3")
        dsv_train = ds[0:3000]
        dsv_val = ds[5000:]
        dsf_train = dsv_train.filter(only_frontal)
        dsf_val = dsv_val.filter(only_frontal)
        print("filtering completed")
        # get train/dev sample counts
        train_counts, train_pos_counts = get_sample_counts(dsf_train, class_names)
        dev_counts, _ = get_sample_counts(dsf_val, class_names)

        # compute steps
        if train_steps == "auto":
            train_steps = int(train_counts / batch_size)
        else:
            try:
                train_steps = int(train_steps)
            except ValueError:
                raise ValueError(
                    f"""
                train_steps: {train_steps} is invalid,
                please use 'auto' or integer.
                """
                )
        print(f"** train_steps: {train_steps} **")

        if validation_steps == "auto":
            validation_steps = int(dev_counts / batch_size)
        else:
            try:
                validation_steps = int(validation_steps)
            except ValueError:
                raise ValueError(
                    f"""
                validation_steps: {validation_steps} is invalid,
                please use 'auto' or integer.
                """
                )
        print(f"** validation_steps: {validation_steps} **")

        # compute class weights
        print("** compute class weights from training data **")
        class_weights = get_class_weights(
            train_counts,
            train_pos_counts,
            multiply=positive_weights_multiply,
        )
        print("** class_weights **")
        print(class_weights)

        print("** load model **")
        if use_trained_model_weights:
            if use_best_weights:
                model_weights_file = os.path.join(
                    output_dir, f"best_{output_weights_name}"
                )
            else:
                model_weights_file = os.path.join(output_dir, output_weights_name)
        else:
            model_weights_file = None

        model_factory = ModelFactory()
        model = model_factory.get_model(
            class_names,
            model_name=base_model_name,
            use_base_weights=use_base_model_weights,
            weights_path=model_weights_file,
            input_shape=(image_dimension, image_dimension, 3),
        )

        if show_model_summary:
            print(model.summary())

        tds_train = dsf_train.to_tensorflow(
            key_list=["image", "label_chexpert", "viewPosition"])
        tds_train = tds_train.map(to_model_fit)
        tds_train = tds_train.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        tds_val = dsf_val.to_tensorflow(
            key_list=["image", "label_chexpert", "viewPosition"])
        tds_val = tds_val.map(to_model_fit)
        tds_val = tds_val.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        print(f"Train data length: {len(dsf_train)}")
        print(f"Val data length: {len(dsf_val)}")

        output_weights_path = os.path.join(output_dir, output_weights_name)
        print(f"** set output weights path to: {output_weights_path} **")
        optimizer = Adam(lr=initial_learning_rate)
        print("** check multiple gpu availability **")
        strategy = tf.distribute.MirroredStrategy()
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        gpus = strategy.num_replicas_in_sync
        if gpus > 1:
            print(f"** multi_gpu_model is used! gpus={gpus} **")
            with strategy.scope():
                model_train = model_factory.get_model(
                    class_names,
                    model_name=base_model_name,
                    use_base_weights=use_base_model_weights,
                    weights_path=model_weights_file,
                    input_shape=(image_dimension, image_dimension, 3),
                )
                model_train.compile(optimizer=optimizer, loss="binary_crossentropy")
        else:
            model_train = model
            model_train.compile(optimizer=optimizer, loss="binary_crossentropy")

        auroc = MultipleClassAUROC(
            sequence=tds_val,
            class_names=class_names,
            weights_path=output_weights_path,
            stats=training_stats,
            workers=generator_workers,
        )
        callbacks = [
            # checkpoint,
            TensorBoard(
                log_dir=os.path.join(output_dir, "logs"), batch_size=batch_size
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.1,
                patience=patience_reduce_lr,
                verbose=1,
                mode="min",
                min_lr=min_lr,
            ),
            # auroc,
        ]
        print("** start training **")
        history = model_train.fit(
            x=dummy_train.repeat(),
            steps_per_epoch=train_steps,
            epochs=epochs,
            validation_data=dummy_val.repeat(),
            validation_steps=validation_steps,
            callbacks=callbacks,
            workers=generator_workers,
            shuffle=False,
        )

        # dump history
        print("** dump history **")
        with open(os.path.join(output_dir, "history.pkl"), "wb") as f:
            pickle.dump(
                {
                    "history": history.history,
                    "auroc": auroc.aurocs,
                },
                f,
            )
        print("** done! **")

    finally:
        os.remove(running_flag_file)


if __name__ == "__main__":
    main()
