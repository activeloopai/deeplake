import os
import posixpath

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tensorflow as tf
import tensorflow_datasets as tfds

import hub
import numpy as np
import requests


DATASETS_DIR = os.path.abspath(".datasets")

if not os.path.isdir(DATASETS_DIR):
    os.mkdir(DATASETS_DIR)


# Add more datasets from: https://www.tensorflow.org/datasets/catalog/overview#all_datasets
datasets = ["coil100", "cifar10", "cifar100", "mnist", "emnist", "fashion_mnist"]


def get_hub_ds_path(dataset: str) -> str:
    # TODO: issue with os.path.join on Windows
    return posixpath.join(DATASETS_DIR, dataset)


def download(dataset: str) -> hub.Dataset:
    data = tfds.as_numpy(
        tfds.load(dataset, split="train")
        .batch(64)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    hub_path = get_hub_ds_path(dataset)
    ds = hub.Dataset(hub_path)
    hub.Dataset.delete(ds)
    ds = hub.Dataset(hub_path)
    with ds:
        for sample in data:
            for col in sample:
                if col not in ds.tensors:
                    ds.create_tensor(col)
                ds[col].append(sample[col])
    return ds
