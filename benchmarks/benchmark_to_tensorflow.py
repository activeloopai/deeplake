from time import time
import tensorflow as tf

import hub
from hub.utils import Timer


def test():
    tf2hub_ds = hub.Dataset.from_tfds('Cifar10', split='train', scheduler='threaded', workers=8)

    res_ds = tf2hub_ds.store("./data/test/cifar/train")
    hub_s3_ds = hub.Dataset(
        url="./data/test/cifar/train", cache=False, storage_cache=False
    )
    for key, value in hub_s3_ds._tensors.items():
        print(key, value.shape, value.chunks)
    hub_s3_ds = hub_s3_ds.to_tensorflow()
    hub_s3_ds = hub_s3_ds.batch(10)
    hub_s3_ds = hub_s3_ds.prefetch(tf.data.AUTOTUNE)
    with Timer("Time"):
        counter = 0
        t0 = time()
        for i, b in enumerate(hub_s3_ds):
            x, y = b["image"], b["label"]
            counter += 100
            t1 = time()
            print(counter, f"dt: {t1 - t0}")
            t0 = t1


if __name__ == "__main__":
    test()
