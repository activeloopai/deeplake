from hub.api.dataset import Dataset
import numpy as np
from hub.integrations.pytorch_old import dataset_to_pytorch
import pytest
from hub.util.check_installation import pytorch_installed
from hub.core.tests.common import parametrize_all_dataset_storages


requires_torch = pytest.mark.skipif(not pytorch_installed(), reason="requires pytorch to be installed")


@requires_torch
@parametrize_all_dataset_storages
def test_pytorch_small(ds):
    import torch

    ds.create_tensor("image")
    ds.image.extend(np.array([i * np.ones((300, 300)) for i in range(256)]))
    ds.create_tensor("image2")
    ds.image2.extend(np.array([i * np.ones((100, 100)) for i in range(256)]))
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
        assert (batch["image2"].numpy() == i * np.ones((1, 100, 100))).all()
    ds.delete()


@requires_torch
@parametrize_all_dataset_storages
def test_pytorch_large(ds):
    import torch

    ds.create_tensor("image")
    arr = np.array(
        [
            np.ones((4096, 4096)),
            2 * np.ones((4096, 4096)),
            3 * np.ones((4096, 4096)),
            4 * np.ones((4096, 4096)),
        ]
    )
    ds.image.extend(arr)
    ds.create_tensor("classlabel")
    ds.classlabel.extend(np.array(range(10)))
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
        assert (batch["classlabel"].numpy() == (i) * np.ones((1,))).all()
    ds.delete()


@requires_torch
@parametrize_all_dataset_storages
def test_pytorch_small_old(ds):
    import torch

    ds.create_tensor("image")
    ds.image.extend(np.array([i * np.ones((300, 300)) for i in range(256)]))
    ds.create_tensor("image2")
    ds.image2.extend(np.array([i * np.ones((100, 100)) for i in range(256)]))
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
        assert (batch["image2"].numpy() == i * np.ones((1, 100, 100))).all()
    ds.delete()


@requires_torch
@parametrize_all_dataset_storages
def test_pytorch_large_old(ds):
    import torch

    ds.create_tensor("image")
    arr = np.array(
        [
            np.ones((4096, 4096)),
            2 * np.ones((4096, 4096)),
            3 * np.ones((4096, 4096)),
            4 * np.ones((4096, 4096)),
        ]
    )
    ds.image.extend(arr)
    ds.create_tensor("classlabel")
    ds.classlabel.extend(np.array([i for i in range(10)]))
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
        assert (batch["classlabel"].numpy() == (i) * np.ones((1,))).all()
    ds.delete()
