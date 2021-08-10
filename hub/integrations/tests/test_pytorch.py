from hub.constants import KB
from hub.util.remove_cache import get_base_storage
import pickle
import pytest
from hub.util.remove_cache import get_base_storage
from hub.util.exceptions import DatasetUnsupportedPytorch
from hub.core.storage.memory import MemoryProvider
import hub
import numpy as np

from hub.integrations.pytorch.pytorch_old import dataset_to_pytorch
from hub.util.check_installation import requires_torch
from hub.tests.dataset_fixtures import enabled_datasets
from hub.core.dataset import Dataset


# ensure tests have multiple chunks without a ton of data
PYTORCH_TESTS_MAX_CHUNK_SIZE = 2 * KB


def to_tuple(sample):
    return sample["image"], sample["image2"]


@requires_torch
@enabled_datasets
def test_pytorch_small(ds):
    with ds:
        ds.create_tensor("image", max_chunk_size=PYTORCH_TESTS_MAX_CHUNK_SIZE)
        ds.image.extend(np.array([i * np.ones((10, 10)) for i in range(16)]))
        ds.create_tensor("image2", max_chunk_size=PYTORCH_TESTS_MAX_CHUNK_SIZE)
        ds.image2.extend(np.array([i * np.ones((12, 12)) for i in range(16)]))

    if isinstance(get_base_storage(ds.storage), MemoryProvider):
        with pytest.raises(DatasetUnsupportedPytorch):
            dl = ds.pytorch(num_workers=2)
        return

    dl = ds.pytorch(num_workers=2, batch_size=1)

    for i, batch in enumerate(dl):
        np.testing.assert_array_equal(batch["image"].numpy(), i * np.ones((1, 10, 10)))
        np.testing.assert_array_equal(batch["image2"].numpy(), i * np.ones((1, 12, 12)))

    sub_ds = ds[5:]

    sub_dl = sub_ds.pytorch(num_workers=2)

    for i, batch in enumerate(sub_dl):
        np.testing.assert_array_equal(
            batch["image"].numpy(), (5 + i) * np.ones((1, 10, 10))
        )
        np.testing.assert_array_equal(
            batch["image2"].numpy(), (5 + i) * np.ones((1, 12, 12))
        )

    sub_ds2 = ds[8:12]

    sub_dl2 = sub_ds2.pytorch(num_workers=2, batch_size=1)

    for i, batch in enumerate(sub_dl2):
        np.testing.assert_array_equal(
            batch["image"].numpy(), (8 + i) * np.ones((1, 10, 10))
        )
        np.testing.assert_array_equal(
            batch["image2"].numpy(), (8 + i) * np.ones((1, 12, 12))
        )

    sub_ds3 = ds[:5]

    sub_dl3 = sub_ds3.pytorch(num_workers=2, batch_size=1)

    for i, batch in enumerate(sub_dl3):
        np.testing.assert_array_equal(
            batch["image"].numpy(), (i) * np.ones((1, 10, 10))
        )
        np.testing.assert_array_equal(
            batch["image2"].numpy(), (i) * np.ones((1, 12, 12))
        )


@requires_torch
@enabled_datasets
def test_pytorch_transform(ds):
    with ds:
        ds.create_tensor("image", max_chunk_size=PYTORCH_TESTS_MAX_CHUNK_SIZE)
        ds.image.extend(np.array([i * np.ones((10, 10)) for i in range(256)]))
        ds.create_tensor("image2", max_chunk_size=PYTORCH_TESTS_MAX_CHUNK_SIZE)
        ds.image2.extend(np.array([i * np.ones((12, 12)) for i in range(256)]))

    if isinstance(get_base_storage(ds.storage), MemoryProvider):
        with pytest.raises(DatasetUnsupportedPytorch):
            dl = ds.pytorch(num_workers=2)
        return

    dl = ds.pytorch(num_workers=2, transform=to_tuple, batch_size=1)

    for i, batch in enumerate(dl):
        actual_image = batch[0].numpy()
        expected_image = i * np.ones((1, 10, 10))
        actual_image2 = batch[1].numpy()
        expected_image2 = i * np.ones((1, 12, 12))
        np.testing.assert_array_equal(actual_image, expected_image)
        np.testing.assert_array_equal(actual_image2, expected_image2)


@requires_torch
@enabled_datasets
def test_pytorch_with_compression(ds: Dataset):
    # TODO: chunk-wise compression for labels (right now they are uncompressed)
    with ds:
        images = ds.create_tensor(
            "images",
            htype="image",
            sample_compression="png",
            max_chunk_size=PYTORCH_TESTS_MAX_CHUNK_SIZE,
        )
        labels = ds.create_tensor(
            "labels", htype="class_label", max_chunk_size=PYTORCH_TESTS_MAX_CHUNK_SIZE
        )

        assert images.meta.sample_compression == "png"

        images.extend(np.ones((16, 12, 12, 3), dtype="uint8"))
        labels.extend(np.ones((16, 1), dtype="uint32"))

    if isinstance(get_base_storage(ds.storage), MemoryProvider):
        with pytest.raises(DatasetUnsupportedPytorch):
            dl = ds.pytorch(num_workers=2)
        return

    dl = ds.pytorch(num_workers=2, batch_size=1)

    for batch in dl:
        X = batch["images"].numpy()
        T = batch["labels"].numpy()
        assert X.shape == (1, 12, 12, 3)
        assert T.shape == (1, 1)


@requires_torch
@enabled_datasets
def test_pytorch_small_old(ds):
    with ds:
        ds.create_tensor("image", max_chunk_size=PYTORCH_TESTS_MAX_CHUNK_SIZE)
        ds.image.extend(np.array([i * np.ones((10, 10)) for i in range(256)]))
        ds.create_tensor("image2", max_chunk_size=PYTORCH_TESTS_MAX_CHUNK_SIZE)
        ds.image2.extend(np.array([i * np.ones((12, 12)) for i in range(256)]))

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
        np.testing.assert_array_equal(batch["image"].numpy(), i * np.ones((1, 10, 10)))
        np.testing.assert_array_equal(batch["image2"].numpy(), i * np.ones((1, 12, 12)))


@requires_torch
@enabled_datasets
def test_custom_tensor_order(ds):
    with ds:
        tensors = ["a", "b", "c", "d"]
        for t in tensors:
            ds.create_tensor(t, max_chunk_size=PYTORCH_TESTS_MAX_CHUNK_SIZE)
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


@requires_torch
def test_readonly(local_ds):
    path = local_ds.path

    local_ds.create_tensor("images", max_chunk_size=PYTORCH_TESTS_MAX_CHUNK_SIZE)
    local_ds.create_tensor("labels", max_chunk_size=PYTORCH_TESTS_MAX_CHUNK_SIZE)
    local_ds.images.extend(np.ones((10, 28, 28)))
    local_ds.labels.extend(np.ones(10))

    del local_ds

    local_ds = hub.dataset(path)
    local_ds.mode = "r"

    # no need to check input, only care that readonly works
    for sample in local_ds.pytorch():
        pass
