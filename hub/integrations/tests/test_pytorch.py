from hub.api.dataset import Dataset
import torch
import numpy as np
from hub.integrations.pytorch_old import dataset_to_pytorch


def test_pytorch_small():
    root = "./test/pytorch"
    ds = Dataset(root)
    ds["image"] = np.array([i * np.ones((300, 300)) for i in range(256)])
    ds.flush()

    ptds = ds.pytorch(workers=2)

    # always use num_workers=0, when using hub workers
    dl = torch.utils.data.DataLoader(
        ptds,
        batch_size=1,
        num_workers=0,
    )
    for i, batch in enumerate(dl):
        assert (batch["image"].numpy() == i * np.ones((1, 300, 300))).all()
    ds.delete()


def test_pytorch_large():
    root = "./test/pytorch"
    ds = Dataset(root)
    ds["image"] = np.array(
        [
            np.ones((4096, 4096)),
            2 * np.ones((4096, 4096)),
            3 * np.ones((4096, 4096)),
            4 * np.ones((4096, 4096)),
        ]
    )
    ds.flush()

    ptds = ds.pytorch(workers=2)

    # always use num_workers=0, when using hub workers
    dl = torch.utils.data.DataLoader(
        ptds,
        batch_size=1,
        num_workers=0,
    )
    for i, batch in enumerate(dl):
        assert (batch["image"].numpy() == (i + 1) * np.ones((1, 4096, 4096))).all()
    ds.delete()


def test_pytorch_small_old():
    root = "./test/pytorch_old"
    ds = Dataset(root)
    ds["image"] = np.array([i * np.ones((300, 300)) for i in range(256)])
    ds.flush()

    # .pytorch will automatically switch depending on version, this syntax is being used to ensure testing of old code on Python 3.8
    ptds = dataset_to_pytorch(ds, workers=2)
    dl = torch.utils.data.DataLoader(
        ptds,
        batch_size=1,
        num_workers=0,
    )
    for i, batch in enumerate(dl):
        assert (batch["image"].numpy() == i * np.ones((1, 300, 300))).all()
    ds.delete()


def test_pytorch_large_old():
    root = "./test/pytorch_old"
    ds = Dataset(root)
    ds["image"] = np.array(
        [
            np.ones((4096, 4096)),
            2 * np.ones((4096, 4096)),
            3 * np.ones((4096, 4096)),
            4 * np.ones((4096, 4096)),
        ]
    )
    ds.flush()

    # .pytorch will automatically switch depending on version, this syntax is being used to ensure testing of old code on Python 3.8
    ptds = dataset_to_pytorch(ds, workers=2)
    dl = torch.utils.data.DataLoader(
        ptds,
        batch_size=1,
        num_workers=0,
    )
    for i, batch in enumerate(dl):
        assert (batch["image"].numpy() == (i + 1) * np.ones((1, 4096, 4096))).all()
    ds.delete()
