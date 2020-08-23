import argparse
import os
import pickle
import numpy as np
from PIL import Image

from hub.collections import dataset, tensor

import tensorflow as tf
import tensorflow_datasets as tfds


def is_equal(lds, rds):
    output = True
    for lex, rex in zip(lds, rds):
        comparsion = lex['data'].numpy() == rex['data'].numpy()
        output *= comparsion.all()
    return output


def check_one(name):
    ds_tf = tfds.load(
        name,
        as_supervised=True,
        split="train",
        shuffle_files=True,
#        with_info=True
    )
    ds_hb = dataset.from_tensorflow(ds_tf)
    stored = ds_hb.store(f"./tmp/{name}")
    stored_tf = stored.to_tensorflow()
    stored_pt = stored.to_pytorch()
    output = is_equal(stored_tf, stored_pt)
    return output


def check_many(names):
    output = []
    for name in names:
        print(name)
        output += [check_one(name)]
    return output


def main():
    #names = tfds.list_builders()
    names = ['mnist', 'cifar10', 'imagenet_resized/8x8']
    results = check_many(names)
    print(results)


if __name__ == "__main__":
    main()
