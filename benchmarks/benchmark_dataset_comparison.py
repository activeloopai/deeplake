import torch
import torchvision
from torchvision import transforms
import tensorflow
import tensorflow_datasets as tfds

from hub import Dataset
from hub.utils import Timer
import os

BATCH_SIZE = 16
PREFETCH_SIZE = 4

ROOT = '.'

DATASET_INFO = [
    {
        'name': 'mnist',
        'pytorch_name': 'MNIST',
        'hub_name': 'activeloop/mnist',
        'split': 'train+test'
    },
    {
        'name': 'places365_small',
        'pytorch_name': 'Places365',
        'hub_name': 'hydp/places365_small_train',
        'split': 'train',
        'kwargs': {'small': True}
    }
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


def prepare_torch_dataset(dataset_info):

    split = dataset_info['split'].split('+')
    trans = transforms.Compose([transforms.ToTensor()])
    dset = getattr(torchvision.datasets, dataset_info['pytorch_name'])
    kwargs = dataset_info.get('kwargs', {})
    if 'train' in split:
        dset = dset(ROOT, train=True, transform=trans, **kwargs)
    else:
        dset = None
    if 'test' in split:
        test_dset = dset(ROOT, train=False, transform=trans, **kwargs)
    else:
        test_dset = None
    if len(split) > 1:
        dset = torch.utils.data.ConcatDataset([dset, test_dset])
    return dset if dset else test_dset


def time_iter_hub_local_pytorch(
    dataset_info,
    batch_size=BATCH_SIZE, prefetch_factor=PREFETCH_SIZE, process=None
):
    mnist = prepare_torch_dataset(dataset_info)
    path = os.path.join(ROOT, 'hub-data', 'torch')
    Dataset.from_pytorch(HubAdapter(mnist)).store(path)
    dset = Dataset(path, cache=False, storage_cache=False, mode="r")

    loader = torch.utils.data.DataLoader(
        dset.to_pytorch(),
        batch_size=batch_size,
        prefetch_factor=prefetch_factor,
        num_workers=1,
    )

    with Timer("Hub_local-PT"):
        for image, label in loader:
            if process is not None:
                process(image, label)


def time_iter_hub_remote_pytorch(
    dataset_info,
    batch_size=BATCH_SIZE, prefetch_factor=PREFETCH_SIZE, process=None
):
    dset = Dataset(
        dataset_info['hub_name'], cache=False, storage_cache=False, mode="r")
    loader = torch.utils.data.DataLoader(
        dset.to_pytorch(),
        batch_size=batch_size,
        prefetch_factor=prefetch_factor,
        num_workers=1
    )

    with Timer("Hub_remote-PT"):
        for image, label in loader:
            if process is not None:
                process(image, label)


def time_iter_pytorch(
    dataset_info,
    batch_size=BATCH_SIZE, prefetch_factor=PREFETCH_SIZE, process=None
):
    dset = prepare_torch_dataset(dataset_info)

    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size,
        prefetch_factor=prefetch_factor,
        num_workers=1)

    with Timer("PyTorch"):
        for image, label in loader:
            if process is not None:
                process(image, label)


def time_iter_hub_local_tensorflow(
    dataset_info,
    batch_size=BATCH_SIZE, prefetch_factor=PREFETCH_SIZE, process=None
):
    dset = Dataset.from_tfds(dataset_info['name'], split=dataset_info['split'])
    path = os.path.join(ROOT, 'hub-data', 'tf')
    dset.store(path)
    dset = Dataset(path, cache=False, storage_cache=False, mode="r")
    loader = dset.to_tensorflow().batch(batch_size).prefetch(prefetch_factor)

    with Timer("Hub_local-TF"):
        for batch in loader:
            image = batch["image"]
            label = batch["label"]
            if process is not None:
                process(image, label)


def time_iter_hub_remote_tensorflow(
    dataset_info,
    batch_size=BATCH_SIZE,
    prefetch_factor=PREFETCH_SIZE,
    process=None
):
    dset = Dataset(
        dataset_info['hub_name'], cache=False, storage_cache=False, mode="r")
    loader = dset.to_tensorflow().batch(batch_size).prefetch(prefetch_factor)

    with Timer("Hub_remote-TF"):
        for batch in loader:
            image = batch["image"]
            label = batch["label"]
            if process is not None:
                process(image, label)


def time_iter_tensorflow(
    dataset_info,
    batch_size=BATCH_SIZE, prefetch_factor=PREFETCH_SIZE, process=None
):
    loader = tfds.load(dataset_info['name'], split=dataset_info['split'])

    with Timer("Tensorflow"):
        for batch in loader:
            image = batch["image"]
            label = batch["label"]
            if process is not None:
                process(image, label)


if __name__ == "__main__":
    for i, info in enumerate(DATASET_INFO):
        time_iter_hub_remote_pytorch(info)
        time_iter_hub_local_pytorch(info)
        time_iter_pytorch(info)
        time_iter_hub_remote_tensorflow(info)
        time_iter_hub_local_tensorflow(info)
        time_iter_tensorflow(info)
