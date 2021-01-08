from time import time

import torchvision
import torch
import numpy as np

import hub
from hub.utils import Timer


class HubAdapter2(torch.utils.data.Dataset):
    def __init__(self, ods):
        self.ds = ods

    def __len__(self):
        return min(len(self.ds), 1000 * 1000)

    @property
    def shape(self):
        return (self.ds.__len__(), None, None, None)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, index):
        x, y = self.ds.__getitem__(index)
        res = {"label": y}
        return res


def test():
    tv_cifar_ds = torchvision.datasets.CIFAR10(".", download=True)

    hub_cifar = HubAdapter2(tv_cifar_ds)

    pt2hb_ds = hub.Dataset.from_pytorch(hub_cifar, scheduler="threaded", workers=8)
    res_ds = pt2hb_ds.store("./data/test/cifar/train")
    hub_s3_ds = hub.Dataset(
        url="./data/test/cifar/train", cache=False, storage_cache=False
    )
    for key, value in hub_s3_ds._tensors.items():
        print(key, value.shape, value.chunks)
    hub_s3_ds = hub_s3_ds.to_pytorch()
    dl = torch.utils.data.DataLoader(hub_s3_ds, batch_size=10, num_workers=0)
    with Timer("Time"):
        counter = 0
        t0 = time()
        for i, b in enumerate(dl):
            x, y = b["label"], b["label"]
            counter += 100
            t1 = time()
            print(counter, f"dt: {t1 - t0}")
            t0 = t1


if __name__ == "__main__":
    test()
