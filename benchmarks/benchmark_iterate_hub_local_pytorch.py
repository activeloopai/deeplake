import torchvision
from torchvision import transforms
import torch
import os

from hub_v1 import Dataset


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


def benchmark_iterate_hub_local_pytorch_setup(
    dataset_name, dataset_split, batch_size, prefetch_factor, num_workers=1
):
    trans = transforms.Compose([transforms.ToTensor()])
    data_path = os.path.join(".", "torch_data")
    dset_type = getattr(torchvision.datasets, dataset_name)
    path = os.path.join(".", "hub_data", "tfds")
    dset = dset_type(
        data_path,
        transform=trans,
        train=(False if "test" in dataset_split else None),
        download=True,
    )

    Dataset.from_pytorch(HubAdapter(dset)).store(path)
    dset = Dataset(path, cache=False, storage_cache=False, mode="r")

    loader = torch.utils.data.DataLoader(
        dset.to_pytorch(),
        batch_size=batch_size,
        prefetch_factor=prefetch_factor,
        num_workers=num_workers,
    )

    return (loader,)


def benchmark_iterate_hub_local_pytorch_run(params):
    (loader,) = params
    for _ in loader:
        pass
