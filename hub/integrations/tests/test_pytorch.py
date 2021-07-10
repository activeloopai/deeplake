import sys
import pickle
import pytest
from hub.util.remove_cache import get_base_storage
from hub.util.exceptions import DatasetUnsupportedPytorch
from hub.core.storage.memory import MemoryProvider
from hub.constants import UNCOMPRESSED
from hub.api.dataset import Dataset
import numpy as np

from hub.integrations.pytorch_old import dataset_to_pytorch
from hub.util.check_installation import requires_torch
from hub.core.tests.common import parametrize_all_dataset_storages


def to_tuple(sample):
    return sample["image"], sample["image2"]


@requires_torch
@parametrize_all_dataset_storages
def test_pytorch_small(ds):
    with ds:
        ds.create_tensor("image")
        ds.image.extend(np.array([i * np.ones((300, 300)) for i in range(256)]))
        ds.create_tensor("image2")
        ds.image2.extend(np.array([i * np.ones((100, 100)) for i in range(256)]))

    if isinstance(get_base_storage(ds.storage), MemoryProvider):
        with pytest.raises(DatasetUnsupportedPytorch):
            dl = ds.pytorch(num_workers=2)
        return

    dl = ds.pytorch(num_workers=2, batch_size=1)

    for i, batch in enumerate(dl):
        np.testing.assert_array_equal(
            batch["image"].numpy(), i * np.ones((1, 300, 300))
        )
        np.testing.assert_array_equal(
            batch["image2"].numpy(), i * np.ones((1, 100, 100))
        )

    sub_ds = ds[50:]

    sub_dl = sub_ds.pytorch(num_workers=2)

    for i, batch in enumerate(sub_dl):
        np.testing.assert_array_equal(
            batch["image"].numpy(), (50 + i) * np.ones((1, 300, 300))
        )
        np.testing.assert_array_equal(
            batch["image2"].numpy(), (50 + i) * np.ones((1, 100, 100))
        )

    sub_ds2 = ds[30:100]

    sub_dl2 = sub_ds2.pytorch(num_workers=2, batch_size=1)

    for i, batch in enumerate(sub_dl2):
        np.testing.assert_array_equal(
            batch["image"].numpy(), (30 + i) * np.ones((1, 300, 300))
        )
        np.testing.assert_array_equal(
            batch["image2"].numpy(), (30 + i) * np.ones((1, 100, 100))
        )

    sub_ds3 = ds[:100]

    sub_dl3 = sub_ds3.pytorch(num_workers=2, batch_size=1)

    for i, batch in enumerate(sub_dl3):
        np.testing.assert_array_equal(
            batch["image"].numpy(), (i) * np.ones((1, 300, 300))
        )
        np.testing.assert_array_equal(
            batch["image2"].numpy(), (i) * np.ones((1, 100, 100))
        )


@requires_torch
@parametrize_all_dataset_storages
def test_pytorch_transform(ds):
    with ds:
        ds.create_tensor("image")
        ds.image.extend(np.array([i * np.ones((300, 300)) for i in range(256)]))
        ds.create_tensor("image2")
        ds.image2.extend(np.array([i * np.ones((100, 100)) for i in range(256)]))

    if isinstance(get_base_storage(ds.storage), MemoryProvider):
        with pytest.raises(DatasetUnsupportedPytorch):
            dl = ds.pytorch(num_workers=2)
        return

    dl = ds.pytorch(num_workers=2, transform=to_tuple, batch_size=1)

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
    # TODO: chunk-wise compression for labels (right now they are uncompressed)
    with ds:
        images = ds.create_tensor("images", htype="image")
        labels = ds.create_tensor("labels", htype="class_label")

        images.extend(np.ones((16, 100, 100, 3), dtype="uint8"))
        labels.extend(np.ones((16, 1), dtype="uint32"))

    if isinstance(get_base_storage(ds.storage), MemoryProvider):
        with pytest.raises(DatasetUnsupportedPytorch):
            dl = ds.pytorch(num_workers=2)
        return

    dl = ds.pytorch(num_workers=2, batch_size=1)

    for batch in dl:
        X = batch["images"].numpy()
        T = batch["labels"].numpy()
        assert X.shape == (1, 100, 100, 3)
        assert T.shape == (1, 1)


@requires_torch
@parametrize_all_dataset_storages
def test_pytorch_small_old(ds):
    with ds:
        ds.create_tensor("image")
        ds.image.extend(np.array([i * np.ones((300, 300)) for i in range(256)]))
        ds.create_tensor("image2")
        ds.image2.extend(np.array([i * np.ones((100, 100)) for i in range(256)]))

    if isinstance(get_base_storage(ds.storage), MemoryProvider):
        with pytest.raises(DatasetUnsupportedPytorch):
            dl = dataset_to_pytorch(
                ds, num_workers=2, batch_size=1, python_version_warning=False
            )
        return

    # .pytorch will automatically switch depending on version, this syntax is being used to ensure testing of old code on Python 3.8
    dl = dataset_to_pytorch(
        ds, num_workers=2, batch_size=1, python_version_warning=False
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
# @pytest.mark.skip(reason="future")
def test_custom_tensor_order(ds):
    with ds:
        tensors = ["a", "b", "c", "d"]
        for t in tensors:
            ds.create_tensor(t)
            ds[t].extend(np.random.random((3, 4, 5)))

    if isinstance(get_base_storage(ds.storage), MemoryProvider):
        with pytest.raises(DatasetUnsupportedPytorch):
            dl = ds.pytorch(num_workers=2)
        return

    dl_new = ds.pytorch(num_workers=2, tensors=["c", "d", "a"])
    dl_old = dataset_to_pytorch(
        ds, num_workers=2, tensors=["c", "d", "a"], python_version_warning=False
    )
    for dl in [dl_new, dl_old]:
        for i, batch in enumerate(dl):
            c1, d1, a1 = batch
            a2 = batch["a"]
            c2 = batch["c"]
            d2 = batch["d"]
            assert "b" not in batch
            np.testing.assert_array_equal(a1, a2)
            np.testing.assert_array_equal(c1, c2)
            np.testing.assert_array_equal(d1, d2)
            np.testing.assert_array_equal(a1[0], ds.a.numpy()[i])
            np.testing.assert_array_equal(c1[0], ds.c.numpy()[i])
            np.testing.assert_array_equal(d1[0], ds.d.numpy()[i])
            batch = pickle.loads(pickle.dumps(batch))
            c1, d1, a1 = batch
            a2 = batch["a"]
            c2 = batch["c"]
            d2 = batch["d"]
            np.testing.assert_array_equal(a1, a2)
            np.testing.assert_array_equal(c1, c2)
            np.testing.assert_array_equal(d1, d2)
            np.testing.assert_array_equal(a1[0], ds.a.numpy()[i])
            np.testing.assert_array_equal(c1[0], ds.c.numpy()[i])
            np.testing.assert_array_equal(d1[0], ds.d.numpy()[i])
