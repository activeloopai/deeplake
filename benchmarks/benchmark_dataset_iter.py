import time
import torch
import tensorflow as tf

from hub import Dataset
from hub.utils import Timer
from hub.schema.features import (
    Primitive,
    Tensor,
    SchemaDict,
    HubSchema,
    featurify,
)

DATASET_NAMES = ['activeloop/mnist', 'activeloop/cifar10_train']

BATCH_SIZES = [1, 16, 128]

PREFETCH_SIZES = [1, 4, 16, 128]

def time_iter_pytorch(dataset_name="activeloop/mnist",
                      batch_size=1,
                      prefetch_factor=0,
                      process=None):

    dset = Dataset(dataset_name, cache=False, storage_cache=False, mode='r')

    loader = torch.utils.data.DataLoader(
            dset.to_pytorch(),
            batch_size=batch_size,
            prefetch_factor=prefetch_factor,
            num_workers=1
            )

    with Timer(f"{dataset_name} PyTorch prefetch {prefetch_factor:03} in batches of {batch_size:03}"):
        for idx, (image, label) in enumerate(loader):
            if process is not None:
                process(idx, image, label)


def time_iter_tensorflow(dataset_name="activeloop/mnist",
                         batch_size=1,
                         prefetch_factor=0,
                         process=None):

    dset = Dataset(dataset_name, cache=False, storage_cache=False, mode='r')

    loader = dset.to_tensorflow().batch(batch_size).prefetch(prefetch_factor)

    with Timer(f"{dataset_name} TF prefetch {prefetch_factor:03} in batches of {batch_size:03}"):
        for idx, batch in enumerate(loader):
            image = batch["image"]
            label = batch["label"]
            if process is not None:
                process(idx, image, label)

if __name__ == "__main__":
    for name in DATASET_NAMES:
        for size in BATCH_SIZES:
            for prefetch in PREFETCH_SIZES:
                time_iter_pytorch(name, size, prefetch, None)
                time_iter_tensorflow(name, size, prefetch, None)
