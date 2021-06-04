import numpy as np
import pytest
from hub.api.dataset import Dataset

from hub.core.meta.dataset_meta import read_dataset_meta

from hub.core.tests.common import parametrize_all_dataset_storages


def test_persist_local_flush(local_storage):
    if local_storage is None:
        pytest.skip()

    ds = Dataset(local_storage.root, local_cache_size=512)
    ds.image = np.ones((4, 4096, 4096))
    ds.flush()
    ds_new = Dataset(local_storage.root)
    assert len(ds_new) == 4
    assert ds_new.image.shape == (4096, 4096)
    np.testing.assert_array_equal(ds_new.image.numpy(), np.ones((4, 4096, 4096)))
    ds.delete()


def test_persist_local_clear_cache(local_storage):
    if local_storage is None:
        pytest.skip()

    ds = Dataset(local_storage.root, local_cache_size=512)
    ds.image = np.ones((4, 4096, 4096))
    ds.clear_cache()
    ds_new = Dataset(local_storage.root)
    assert len(ds_new) == 4
    assert ds_new.image.shape == (4096, 4096)
    np.testing.assert_array_equal(ds_new.image.numpy(), np.ones((4, 4096, 4096)))
    ds.delete()


@parametrize_all_dataset_storages
def test_populate_dataset(ds):
    assert read_dataset_meta(ds.storage) == {"tensors": []}
    ds.image = np.ones((4, 28, 28))
    assert read_dataset_meta(ds.storage) == {"tensors": ["image"]}
    assert len(ds) == 4


@parametrize_all_dataset_storages
def test_compute_fixed_tensor(ds):
    ds.image = np.ones((32, 28, 28))
    np.testing.assert_array_equal(ds.image.numpy(), np.ones((32, 28, 28)))


@parametrize_all_dataset_storages
def test_compute_dynamic_tensor(ds):
    ds.image = np.ones((32, 28, 28))
    ds.image.append(np.ones((10, 36, 11)), batched=True)

    expected_list = [*np.ones((32, 28, 28)), *np.ones((10, 36, 11))]
    actual_list = ds.image.numpy(aslist=True)

    for a1, a2 in zip(expected_list, actual_list):
        assert (a1 == a2).all()


@parametrize_all_dataset_storages
def test_compute_tensor_slice(ds):
    ds.image = np.vstack((np.arange(16),) * 8)

    sliced_data = ds.image[2:5].numpy()
    expected_data = np.vstack((np.arange(16),) * 3)
    np.testing.assert_array_equal(sliced_data, expected_data)


@parametrize_all_dataset_storages
def test_iterate_dataset(ds):
    labels = [1, 9, 7, 4]
    ds.image = np.ones((4, 28, 28))
    ds.label = np.asarray(labels).reshape((4, 1))

    for idx, sub_ds in enumerate(ds):
        img = sub_ds.image.numpy()
        label = sub_ds.label.numpy()
        np.testing.assert_array_equal(img, np.ones((28, 28)))
        assert label.shape == (1,)
        assert label == labels[idx]


def test_shape_property(memory_ds):
    memory_ds.image1 = np.ones((32, 28, 28))
    memory_ds.image1.append(np.ones((16, 33, 9)), batched=True)

    shape1 = memory_ds.image1.shape
    assert shape1.lower == (28, 9)
    assert shape1.upper == (33, 28)

    memory_ds.image2 = np.ones((32, 28, 28))
    memory_ds.image2.append(np.ones((28, 28)), batched=False)

    shape2 = memory_ds.image2.shape
    assert shape2.lower == (28, 28)
    assert shape2.upper == (28, 28)
