from hub.util.remove_cache import remove_memory_cache
import pytest
from hub.util.exceptions import DatasetUnsupportedPytorch
from hub.core.storage.memory import MemoryProvider
from hub.tests.common import assert_all_samples_have_expected_compression
from hub.constants import UNCOMPRESSED
from hub.api.dataset import Dataset
import numpy as np

from hub.integrations.pytorch_old import dataset_to_pytorch
from hub.util.check_installation import requires_torch
from hub.core.tests.common import parametrize_all_dataset_storages


@requires_torch
@parametrize_all_dataset_storages
def test_pytorch_small(ds):
    import torch

    with ds:
        ds.create_tensor("image")
        ds.image.extend(np.array([i * np.ones((300, 300)) for i in range(256)]))
        ds.create_tensor("image2")
        ds.image2.extend(np.array([i * np.ones((100, 100)) for i in range(256)]))

    if isinstance(remove_memory_cache(ds.storage), MemoryProvider):
        with pytest.raises(DatasetUnsupportedPytorch):
            ptds = ds.pytorch(workers=2)
        return
    ptds = ds.pytorch(workers=2)

    # always use num_workers=0, when using hub workers
    dl = torch.utils.data.DataLoader(
        ptds,
        batch_size=1,
        num_workers=0,
    )
    for i, batch in enumerate(dl):
        np.testing.assert_array_equal(
            batch["image"].numpy(), i * np.ones((1, 300, 300))
        )
        np.testing.assert_array_equal(
            batch["image2"].numpy(), i * np.ones((1, 100, 100))
        )

    sub_ds = ds[50:]

    sub_ptds = sub_ds.pytorch(workers=2)

    # always use num_workers=0, when using hub workers
    sub_dl = torch.utils.data.DataLoader(
        sub_ptds,
        batch_size=1,
        num_workers=0,
    )

    for i, batch in enumerate(sub_dl):
        np.testing.assert_array_equal(
            batch["image"].numpy(), (50 + i) * np.ones((1, 300, 300))
        )
        np.testing.assert_array_equal(
            batch["image2"].numpy(), (50 + i) * np.ones((1, 100, 100))
        )

    sub_ds2 = ds[30:100]

    sub_ptds2 = sub_ds2.pytorch(workers=2)

    # always use num_workers=0, when using hub workers
    sub_dl2 = torch.utils.data.DataLoader(
        sub_ptds2,
        batch_size=1,
        num_workers=0,
    )

    for i, batch in enumerate(sub_dl2):
        np.testing.assert_array_equal(
            batch["image"].numpy(), (30 + i) * np.ones((1, 300, 300))
        )
        np.testing.assert_array_equal(
            batch["image2"].numpy(), (30 + i) * np.ones((1, 100, 100))
        )

    sub_ds3 = ds[:100]

    sub_ptds3 = sub_ds3.pytorch(workers=2)

    # always use num_workers=0, when using hub workers
    sub_dl3 = torch.utils.data.DataLoader(
        sub_ptds3,
        batch_size=1,
        num_workers=0,
    )

    for i, batch in enumerate(sub_dl3):
        np.testing.assert_array_equal(
            batch["image"].numpy(), (i) * np.ones((1, 300, 300))
        )
        np.testing.assert_array_equal(
            batch["image2"].numpy(), (i) * np.ones((1, 100, 100))
        )


@requires_torch
@parametrize_all_dataset_storages
def test_pytorch_large(ds):
    import torch

    with ds:
        ds.create_tensor("image")
        arr = np.array(
            [
                np.ones((2200, 2200)),
                2 * np.ones((2200, 2200)),
                3 * np.ones((2200, 2200)),
            ]
        )
        ds.image.extend(arr)
        ds.create_tensor("classlabel")
        ds.classlabel.extend(np.array([i for i in range(10)]))

    if isinstance(remove_memory_cache(ds.storage), MemoryProvider):
        with pytest.raises(DatasetUnsupportedPytorch):
            ptds = ds.pytorch(workers=2)
        return

    ptds = ds.pytorch(workers=2)

    # always use num_workers=0, when using hub workers
    dl = torch.utils.data.DataLoader(
        ptds,
        batch_size=1,
        num_workers=0,
    )
    for i, batch in enumerate(dl):
        actual_image = batch["image"].numpy()
        expected_image = (i + 1) * np.ones((1, 2200, 2200))

        actual_label = batch["classlabel"].numpy()
        expected_label = (i) * np.ones((1,))

        np.testing.assert_array_equal(actual_image, expected_image)
        np.testing.assert_array_equal(actual_label, expected_label)


@requires_torch
@parametrize_all_dataset_storages
def test_pytorch_transform(ds):
    import torch

    with ds:
        ds.create_tensor("image")
        ds.image.extend(np.array([i * np.ones((300, 300)) for i in range(256)]))
        ds.create_tensor("image2")
        ds.image2.extend(np.array([i * np.ones((100, 100)) for i in range(256)]))

    def to_tuple(sample):
        return sample["image"], sample["image2"]

    if isinstance(remove_memory_cache(ds.storage), MemoryProvider):
        with pytest.raises(DatasetUnsupportedPytorch):
            ptds = ds.pytorch(workers=2)
        return

    ptds = ds.pytorch(workers=2, transform=to_tuple)
    dl = torch.utils.data.DataLoader(
        ptds,
        batch_size=1,
        num_workers=0,
    )

    for i, batch in enumerate(dl):
        actual_image = batch[0].numpy()
        expected_image = i * np.ones((1, 300, 300))
        actual_image2 = batch[1].numpy()
        expected_image2 = i * np.ones((1, 100, 100))
        np.testing.assert_array_equal(actual_image, expected_image)
        np.testing.assert_array_equal(actual_image2, expected_image2)


@requires_torch
@parametrize_all_dataset_storages
def test_pytorch_with_compression(ds: Dataset):
    import torch

    # TODO: chunk-wise compression for labels (right now they are uncompressed)
    with ds:
        images = ds.create_tensor("images", htype="image")
        labels = ds.create_tensor("labels", htype="class_label")

        images.extend(np.ones((16, 100, 100, 3), dtype="uint8"))
        labels.extend(np.ones((16, 1), dtype="int32"))

    # make sure data is appropriately compressed
    assert images.meta.sample_compression == "png"
    assert labels.meta.sample_compression == UNCOMPRESSED
    assert_all_samples_have_expected_compression(images, ["png"] * 16)
    assert_all_samples_have_expected_compression(labels, [UNCOMPRESSED] * 16)

    if isinstance(remove_memory_cache(ds.storage), MemoryProvider):
        with pytest.raises(DatasetUnsupportedPytorch):
            ptds = ds.pytorch(workers=2)
        return
    ptds = ds.pytorch(workers=2)
    dl = torch.utils.data.DataLoader(ptds, batch_size=1, num_workers=0)

    for batch in dl:
        X = batch["images"].numpy()
        T = batch["labels"].numpy()
        assert X.shape == (1, 100, 100, 3)
        assert T.shape == (1, 1)


@requires_torch
@parametrize_all_dataset_storages
def test_pytorch_small_old(ds):
    import torch

    with ds:
        ds.create_tensor("image")
        ds.image.extend(np.array([i * np.ones((300, 300)) for i in range(256)]))
        ds.create_tensor("image2")
        ds.image2.extend(np.array([i * np.ones((100, 100)) for i in range(256)]))

    # .pytorch will automatically switch depending on version, this syntax is being used to ensure testing of old code on Python 3.8
    ptds = dataset_to_pytorch(ds, workers=2, python_version_warning=False)
    dl = torch.utils.data.DataLoader(
        ptds,
        batch_size=1,
        num_workers=0,
    )
    for i, batch in enumerate(dl):
        np.testing.assert_array_equal(
            batch["image"].numpy(), i * np.ones((1, 300, 300))
        )
        np.testing.assert_array_equal(
            batch["image2"].numpy(), i * np.ones((1, 100, 100))
        )


@requires_torch
@parametrize_all_dataset_storages
def test_pytorch_large_old(ds):
    import torch

    # don't need to test with compression because it uses the API (which is tested for iteration + compression)
    with ds:
        ds.create_tensor("image")
        arr = np.array(
            [
                np.ones((2200, 2200)),
                2 * np.ones((2200, 2200)),
                3 * np.ones((2200, 2200)),
            ],
            dtype="uint8",
        )
        ds.image.extend(arr)
        ds.create_tensor("classlabel")
        ds.classlabel.extend(np.array([i for i in range(10)], dtype="uint32"))

    # .pytorch will automatically switch depending on version, this syntax is being used to ensure testing of old code on Python 3.8
    ptds = dataset_to_pytorch(ds, workers=2, python_version_warning=False)
    dl = torch.utils.data.DataLoader(
        ptds,
        batch_size=1,
        num_workers=0,
    )
    for i, batch in enumerate(dl):
        actual_image = batch["image"].numpy()
        expected_image = (i + 1) * np.ones((1, 2200, 2200))

        actual_label = batch["classlabel"].numpy()
        expected_label = (i) * np.ones((1,))

        np.testing.assert_array_equal(actual_image, expected_image)
        np.testing.assert_array_equal(actual_label, expected_label)
