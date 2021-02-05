import torch
import torchvision
from torchvision import transforms
import tensorflow as tf
import tensorflow_datasets as tfds

from hub import Dataset
from hub.utils import Timer
import os

# import math

BATCH_SIZE = 16
PREFETCH_SIZE = 4
NUM_WORKERS = 1
# CPUS = os.cpu_count()
# NUM_WORKERS = [
#    min(2 ** n, CPUS) for n in range(math.ceil(math.log2(CPUS)) + 1)]

ROOT = "."
S3_PATH = "s3://snark-benchmarks/datasets/Hub/"

DATASET_INFO = [
    {
        "name": "mnist",
        "pytorch_name": "MNIST",
        "hub_name": "activeloop/mnist",
        "s3_name": "mnist",
        "split": "train+test",
    },
    {
        "name": "places365_small",
        "pytorch_name": "Places365",
        "hub_name": "hydp/places365_small_train",
        "s3_name": "places365_small_train",
        "split": "train",
        "kwargs": {"small": True},
    },
]


class HubAdapter(torch.utils.data.Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    @property
    def shape(self):
        return (len(self), None, None, None)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, index):
        img, label = self.ds.__getitem__(index)
        return {"image": img, "label": label}


class Timer(Timer):
    def __init__(self, text):
        super().__init__(text)
        self._text = f"BENCHMARK - {self._text}"


def prepare_torch_dataset(dataset_info):
    split = dataset_info["split"].split("+")
    trans = transforms.Compose([transforms.ToTensor()])
    data_path = "torch_data"
    dset_type = getattr(torchvision.datasets, dataset_info["pytorch_name"])
    kwargs = dataset_info.get("kwargs", {})
    if "train" in split:
        dset = dset_type(
            os.path.join(ROOT, data_path), transform=trans, download=True, **kwargs
        )
    else:
        dset = None
    if "test" in split:
        test_dset = dset_type(
            os.path.join(ROOT, data_path),
            train=False,
            transform=trans,
            download=True,
            **kwargs,
        )
    else:
        test_dset = None
    if len(split) > 1:
        dset = torch.utils.data.ConcatDataset([dset, test_dset])
    return dset if dset else test_dset


def time_iter_hub_local_pytorch(
    dataset_info,
    batch_size=BATCH_SIZE,
    prefetch_factor=PREFETCH_SIZE,
    num_workers=NUM_WORKERS,
    process=None,
):
    mnist = prepare_torch_dataset(dataset_info)
    path = os.path.join(ROOT, "Hub_data", "torch")
    Dataset.from_pytorch(HubAdapter(mnist)).store(path)
    dset = Dataset(path, cache=False, storage_cache=False, mode="r")

    loader = torch.utils.data.DataLoader(
        dset.to_pytorch(),
        batch_size=batch_size,
        prefetch_factor=prefetch_factor,
        num_workers=num_workers,
    )

    with Timer("Hub (local) `.to_pytorch()`"):
        for image, label in loader:
            if process is not None:
                process(image, label)


def time_iter_hub_wasabi_pytorch(
    dataset_info,
    batch_size=BATCH_SIZE,
    prefetch_factor=PREFETCH_SIZE,
    num_workers=NUM_WORKERS,
    process=None,
):
    dset = Dataset(dataset_info["hub_name"], cache=False, storage_cache=False, mode="r")
    loader = torch.utils.data.DataLoader(
        dset.to_pytorch(),
        batch_size=batch_size,
        prefetch_factor=prefetch_factor,
        num_workers=num_workers,
    )

    with Timer("Hub (remote - Wasabi) `.to_pytorch()`"):
        for image, label in loader:
            if process is not None:
                process(image, label)


def time_iter_hub_s3_pytorch(
    dataset_info,
    batch_size=BATCH_SIZE,
    prefetch_factor=PREFETCH_SIZE,
    num_workers=NUM_WORKERS,
    process=None,
):
    dset = Dataset(
        f"{S3_PATH}{dataset_info['s3_name']}",
        cache=False,
        storage_cache=False,
        mode="r",
    )
    loader = torch.utils.data.DataLoader(
        dset.to_pytorch(),
        batch_size=batch_size,
        prefetch_factor=prefetch_factor,
        num_workers=num_workers,
    )

    with Timer("Hub (remote - S3) `.to_pytorch()`"):
        for image, label in loader:
            if process is not None:
                process(image, label)


def time_iter_pytorch(
    dataset_info,
    batch_size=BATCH_SIZE,
    prefetch_factor=PREFETCH_SIZE,
    num_workers=NUM_WORKERS,
    process=None,
):
    dset = prepare_torch_dataset(dataset_info)

    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size,
        prefetch_factor=prefetch_factor,
        num_workers=num_workers,
    )

    with Timer("PyTorch (local, native)"):
        for image, label in loader:
            if process is not None:
                process(image, label)


def time_iter_hub_local_tensorflow(
    dataset_info, batch_size=BATCH_SIZE, prefetch_factor=PREFETCH_SIZE, process=None
):
    dset = Dataset.from_tfds(dataset_info["name"], split=dataset_info["split"])
    path = os.path.join(ROOT, "Hub_data", "tfds")
    dset.store(path)
    dset = Dataset(path, cache=False, storage_cache=False, mode="r")
    loader = dset.to_tensorflow().batch(batch_size).prefetch(prefetch_factor)

    with Timer("Hub (local) `.to_tensorflow()`"):
        for batch in loader:
            image = batch["image"]
            label = batch["label"]
            if process is not None:
                process(image, label)


def time_iter_hub_wasabi_tensorflow(
    dataset_info, batch_size=BATCH_SIZE, prefetch_factor=PREFETCH_SIZE, process=None
):
    dset = Dataset(dataset_info["hub_name"], cache=False, storage_cache=False, mode="r")
    loader = dset.to_tensorflow().batch(batch_size).prefetch(prefetch_factor)

    with Timer("Hub (remote - Wasabi) `.to_tensorflow()`"):
        for batch in loader:
            image = batch["image"]
            label = batch["label"]
            if process is not None:
                process(image, label)


def time_iter_hub_s3_tensorflow(
    dataset_info, batch_size=BATCH_SIZE, prefetch_factor=PREFETCH_SIZE, process=None
):
    dset = Dataset(
        f"{S3_PATH}{dataset_info['s3_name']}",
        cache=False,
        storage_cache=False,
        mode="r",
    )
    loader = dset.to_tensorflow().batch(batch_size).prefetch(prefetch_factor)

    with Timer("Hub (remote - S3) `.to_tensorflow()`"):
        for batch in loader:
            image = batch["image"]
            label = batch["label"]
            if process is not None:
                process(image, label)


def time_iter_tensorflow(
    dataset_info, batch_size=BATCH_SIZE, prefetch_factor=PREFETCH_SIZE, process=None
):
    # turn off auto-splitting and disable multiprocessing
    options = tf.data.Options()
    blockAS = tf.data.experimental.AutoShardPolicy.OFF
    options.experimental_distribute.auto_shard_policy = blockAS
    options.experimental_optimization.autotune_cpu_budget = 1

    loader = tfds.load(dataset_info["name"], split=dataset_info["split"]).with_options(
        options
    )

    with Timer("Tensorflow (local, native - TFDS)"):
        for batch in loader:
            image = batch["image"]
            label = batch["label"]
            if process is not None:
                process(image, label)


if __name__ == "__main__":
    for i, info in enumerate(DATASET_INFO):
        print(f'BENCHMARK DATASET #{i}: {info["name"]}')
        time_iter_hub_wasabi_pytorch(info)
        time_iter_hub_local_pytorch(info)
        time_iter_hub_s3_pytorch(info)
        time_iter_pytorch(info)
        time_iter_hub_wasabi_tensorflow(info)
        time_iter_hub_local_tensorflow(info)
        time_iter_hub_s3_tensorflow(info)
        time_iter_tensorflow(info)
