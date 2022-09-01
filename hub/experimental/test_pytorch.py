from hub.experimental import dataloader
import sys

import hub
import numpy as np
import pytest

from hub.util.remove_cache import get_base_storage
from hub.tests.common import requires_torch, requires_linux
from hub.core.dataset import Dataset
from hub.core.storage import MemoryProvider, GCSProvider
from hub.constants import KB

from hub.tests.dataset_fixtures import enabled_non_gdrive_datasets

try:
    from torch.utils.data._utils.collate import default_collate
except ImportError:
    pass

# ensure tests have multiple chunks without a ton of data
PYTORCH_TESTS_MAX_CHUNK_SIZE = 5 * KB


def double(sample):
    return sample * 2


def to_tuple(sample):
    return sample["image"], sample["image2"]


def reorder_collate(batch):
    x = [((x["a"], x["b"]), x["c"]) for x in batch]
    return default_collate(x)


def dict_to_list(sample):
    return [sample["a"], sample["b"], sample["c"]]


def my_transform_collate(batch):
    x = [(c, a, b) for a, b, c in batch]
    return default_collate(x)


@requires_torch
@enabled_non_gdrive_datasets
@requires_linux
def test_pytorch_small(ds):
    with ds:
        ds.create_tensor("image", max_chunk_size=PYTORCH_TESTS_MAX_CHUNK_SIZE)
        ds.image.extend(([i * np.ones((i + 1, i + 1)) for i in range(16)]))
        ds.commit()
        ds.create_tensor("image2", max_chunk_size=PYTORCH_TESTS_MAX_CHUNK_SIZE)
        ds.image2.extend(np.array([i * np.ones((12, 12)) for i in range(16)]))

    if isinstance(get_base_storage(ds.storage), (MemoryProvider, GCSProvider)):
        with pytest.raises(ValueError):
            dl = dataloader(ds)
        return
    dl = dataloader(ds).batch(1).pytorch(num_workers=2)

    assert len(dl.dataset) == 16

    for _ in range(2):
        for i, batch in enumerate(dl):
            np.testing.assert_array_equal(
                batch["image"].numpy(), i * np.ones((1, i + 1, i + 1))
            )
            np.testing.assert_array_equal(
                batch["image2"].numpy(), i * np.ones((1, 12, 12))
            )

    sub_ds = ds[5:]
    sub_dl = dataloader(sub_ds).pytorch(num_workers=0)

    for i, batch in enumerate(sub_dl):
        np.testing.assert_array_equal(
            batch["image"].numpy(), (5 + i) * np.ones((1, 6 + i, 6 + i))
        )
        np.testing.assert_array_equal(
            batch["image2"].numpy(), (5 + i) * np.ones((1, 12, 12))
        )

    sub_ds2 = ds[8:12]
    sub_dl2 = dataloader(sub_ds2).pytorch(num_workers=0)

    for _ in range(2):
        for i, batch in enumerate(sub_dl2):
            np.testing.assert_array_equal(
                batch["image"].numpy(), (8 + i) * np.ones((1, 9 + i, 9 + i))
            )
            np.testing.assert_array_equal(
                batch["image2"].numpy(), (8 + i) * np.ones((1, 12, 12))
            )

    sub_ds3 = ds[:5]
    sub_dl3 = dataloader(sub_ds3).pytorch(num_workers=0)

    for _ in range(2):
        for i, batch in enumerate(sub_dl3):
            np.testing.assert_array_equal(
                batch["image"].numpy(), (i) * np.ones((1, i + 1, i + 1))
            )
            np.testing.assert_array_equal(
                batch["image2"].numpy(), (i) * np.ones((1, 12, 12))
            )


@requires_torch
@enabled_non_gdrive_datasets
@requires_linux
def test_pytorch_transform(ds):
    with ds:
        ds.create_tensor("image", max_chunk_size=PYTORCH_TESTS_MAX_CHUNK_SIZE)
        ds.image.extend(([i * np.ones((i + 1, i + 1)) for i in range(16)]))
        ds.checkout("alt", create=True)
        ds.create_tensor("image2", max_chunk_size=PYTORCH_TESTS_MAX_CHUNK_SIZE)
        ds.image2.extend(np.array([i * np.ones((12, 12)) for i in range(16)]))

    if isinstance(get_base_storage(ds.storage), (MemoryProvider, GCSProvider)):
        with pytest.raises(ValueError):
            dl = dataloader(ds)
        return

    dl = dataloader(ds).batch(1).transform(to_tuple).pytorch(num_workers=2)

    for _ in range(2):
        for i, batch in enumerate(dl):
            actual_image = batch[0].numpy()
            expected_image = i * np.ones((1, i + 1, i + 1))
            actual_image2 = batch[1].numpy()
            expected_image2 = i * np.ones((1, 12, 12))
            np.testing.assert_array_equal(actual_image, expected_image)
            np.testing.assert_array_equal(actual_image2, expected_image2)


@requires_torch
@enabled_non_gdrive_datasets
@requires_linux
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

    if isinstance(get_base_storage(ds.storage), (MemoryProvider, GCSProvider)):
        with pytest.raises(ValueError):
            dl = dataloader(ds)
        return

    dl = dataloader(ds).pytorch(num_workers=0)

    for _ in range(2):
        for batch in dl:
            X = batch["images"].numpy()
            T = batch["labels"].numpy()
            assert X.shape == (1, 12, 12, 3)
            assert T.shape == (1, 1)


@requires_torch
@requires_linux
def test_readonly_with_two_workers(local_ds):
    local_ds.create_tensor("images", max_chunk_size=PYTORCH_TESTS_MAX_CHUNK_SIZE)
    local_ds.create_tensor("labels", max_chunk_size=PYTORCH_TESTS_MAX_CHUNK_SIZE)
    local_ds.images.extend(np.ones((10, 12, 12)))
    local_ds.labels.extend(np.ones(10))

    base_storage = get_base_storage(local_ds.storage)
    base_storage.flush()
    base_storage.enable_readonly()
    ds = Dataset(storage=local_ds.storage, read_only=True, verbose=False)

    ptds = dataloader(ds).pytorch(num_workers=2)
    # no need to check input, only care that readonly works
    for _ in ptds:
        pass


@requires_torch
@requires_linux
def test_groups(local_ds, compressed_image_paths):
    img1 = hub.read(compressed_image_paths["jpeg"][0])
    img2 = hub.read(compressed_image_paths["png"][0])
    with local_ds:
        local_ds.create_tensor(
            "images/jpegs/cats", htype="image", sample_compression="jpeg"
        )
        local_ds.create_tensor(
            "images/pngs/flowers", htype="image", sample_compression="png"
        )
        for _ in range(10):
            local_ds.images.jpegs.cats.append(img1)
            local_ds.images.pngs.flowers.append(img2)

    another_ds = hub.dataset(local_ds.path)
    dl = dataloader(another_ds).pytorch()
    for i, (cat, flower) in enumerate(dl):
        assert cat[0].shape == another_ds.images.jpegs.cats[i].numpy().shape
        assert flower[0].shape == another_ds.images.pngs.flowers[i].numpy().shape


@requires_torch
@requires_linux
def test_string_tensors(local_ds):
    with local_ds:
        local_ds.create_tensor("strings", htype="text")
        local_ds.strings.extend([f"string{idx}" for idx in range(5)])

    ptds = dataloader(local_ds).pytorch()
    for idx, batch in enumerate(ptds):
        np.testing.assert_array_equal(batch["strings"], f"string{idx}")


@requires_torch
@requires_linux
def test_pytorch_collate(local_ds):
    with local_ds:
        local_ds.create_tensor("a")
        local_ds.create_tensor("b")
        local_ds.create_tensor("c")
        for _ in range(100):
            local_ds.a.append(0)
            local_ds.b.append(1)
            local_ds.c.append(2)

    ptds = (
        dataloader(local_ds)
        .batch(4)
        .pytorch(
            collate_fn=reorder_collate,
        )
    )
    for batch in ptds:
        assert len(batch) == 2
        assert len(batch[0]) == 2
        np.testing.assert_array_equal(batch[0][0], np.array([0, 0, 0, 0]).reshape(4, 1))
        np.testing.assert_array_equal(batch[0][1], np.array([1, 1, 1, 1]).reshape(4, 1))
        np.testing.assert_array_equal(batch[1], np.array([2, 2, 2, 2]).reshape(4, 1))


@requires_torch
@requires_linux
def test_pytorch_transform_collate(local_ds):
    with local_ds:
        local_ds.create_tensor("a")
        local_ds.create_tensor("b")
        local_ds.create_tensor("c")
        for _ in range(100):
            local_ds.a.append(0 * np.ones((300, 300)))
            local_ds.b.append(1 * np.ones((300, 300)))
            local_ds.c.append(2 * np.ones((300, 300)))

    ptds = (
        dataloader(local_ds)
        .batch(4)
        .pytorch(
            collate_fn=my_transform_collate,
        )
        .transform(dict_to_list)
    )
    for batch in ptds:
        assert len(batch) == 3
        for i in range(2):
            assert len(batch[i]) == 4
        np.testing.assert_array_equal(batch[0], 2 * np.ones((4, 300, 300)))
        np.testing.assert_array_equal(batch[1], 0 * np.ones((4, 300, 300)))
        np.testing.assert_array_equal(batch[2], 1 * np.ones((4, 300, 300)))


@requires_torch
@requires_linux
def test_rename(local_ds):
    with local_ds as ds:
        ds.create_tensor("abc")
        ds.create_tensor("blue/green")
        ds.abc.append([1, 2, 3])
        ds.rename_tensor("abc", "xyz")
        ds.rename_group("blue", "red")
        ds["red/green"].append([1, 2, 3, 4])
    loader = dataloader(ds).pytorch()
    for sample in loader:
        assert set(sample.keys()) == {"xyz", "red/green"}
        np.testing.assert_array_equal(np.array(sample["xyz"]), np.array([[1, 2, 3]]))
        np.testing.assert_array_equal(
            np.array(sample["red/green"]), np.array([[1, 2, 3, 4]])
        )


@requires_linux
def test_uneven_iteration(local_ds):
    with local_ds as ds:
        ds.create_tensor("x")
        ds.create_tensor("y")
        ds.x.extend(list(range(5)))
        ds.y.extend(list(range(10)))
    ptds = dataloader(ds).pytorch()
    for i, batch in enumerate(ptds):
        x, y = np.array(batch["x"][0]), np.array(batch["y"][0])
        np.testing.assert_equal(x, i)
        np.testing.assert_equal(y, i)
