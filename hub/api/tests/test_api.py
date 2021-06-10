import numpy as np
import pytest

from hub.api.dataset import Dataset
from hub.core.tests.common import parametrize_all_dataset_storages


def test_persist_local_flush(local_storage):
    if local_storage is None:
        pytest.skip()

    ds = Dataset(local_storage.root, local_cache_size=512)
    ds.create_tensor("image")
    ds.image.extend(np.ones((4, 4096, 4096)))
    ds.flush()

    ds_new = Dataset(local_storage.root)
    assert len(ds_new) == 4

    assert ds_new.image.shape.lower == (4096, 4096)
    assert ds_new.image.shape.upper == (4096, 4096)

    np.testing.assert_array_equal(ds_new.image.numpy(), np.ones((4, 4096, 4096)))
    ds.delete()


def test_persist_local_clear_cache(local_storage):
    if local_storage is None:
        pytest.skip()

    ds = Dataset(local_storage.root, local_cache_size=512)
    ds.create_tensor("image")
    ds.image.extend(np.ones((4, 4096, 4096)))
    ds.clear_cache()
    ds_new = Dataset(local_storage.root)
    assert len(ds_new) == 4

    assert ds_new.image.shape.lower == (4096, 4096)
    assert ds_new.image.shape.upper == (4096, 4096)

    np.testing.assert_array_equal(ds_new.image.numpy(), np.ones((4, 4096, 4096)))
    ds.delete()


@parametrize_all_dataset_storages
def test_populate_dataset(ds):
    assert ds.meta.tensors == []
    ds.create_tensor("image")
    assert len(ds) == 0
    assert len(ds.image) == 0

    ds.image.extend(np.ones((4, 28, 28)))
    assert len(ds) == 4
    assert len(ds.image) == 4

    for _ in range(10):
        ds.image.append(np.ones((28, 28)))
    assert len(ds.image) == 14

    ds.image.extend([np.ones((28, 28)), np.ones((28, 28))])
    assert len(ds.image) == 16

    assert ds.meta.tensors == ["image"]


@parametrize_all_dataset_storages
def test_compute_fixed_tensor(ds):
    ds.create_tensor("image")
    ds.image.extend(np.ones((32, 28, 28)))
    np.testing.assert_array_equal(ds.image.numpy(), np.ones((32, 28, 28)))


@parametrize_all_dataset_storages
def test_compute_dynamic_tensor(ds):
    ds.create_tensor("image")

    a1 = np.ones((32, 28, 28))
    a2 = np.ones((10, 36, 11))
    a3 = np.ones((29, 10))

    image = ds.image

    image.extend(a1)
    image.extend(a2)
    image.append(a3)

    expected_list = [*a1, *a2, a3]
    actual_list = image.numpy(aslist=True)

    for expected, actual in zip(expected_list, actual_list):
        np.testing.assert_array_equal(expected, actual)

    assert image.shape.lower == (28, 10)
    assert image.shape.upper == (36, 28)
    assert image.shape.is_dynamic


@parametrize_all_dataset_storages
def test_compute_tensor_slice(ds):
    ds.create_tensor("image")
    ds.image.extend(np.vstack((np.arange(16),) * 8))

    sliced_data = ds.image[2:5].numpy()
    expected_data = np.vstack((np.arange(16),) * 3)
    np.testing.assert_array_equal(sliced_data, expected_data)


@parametrize_all_dataset_storages
def test_iterate_dataset(ds):
    labels = [1, 9, 7, 4]
    ds.create_tensor("image")
    ds.create_tensor("label")

    ds.image.extend(np.ones((4, 28, 28)))
    ds.label.extend(np.asarray(labels).reshape((4, 1)))

    for idx, sub_ds in enumerate(ds):
        img = sub_ds.image.numpy()
        label = sub_ds.label.numpy()
        np.testing.assert_array_equal(img, np.ones((28, 28)))
        assert label.shape == (1,)
        assert label == labels[idx]


def test_shape_property(memory_ds):
    fixed = memory_ds.create_tensor("fixed_tensor")
    dynamic = memory_ds.create_tensor("dynamic_tensor")

    # dynamic shape property
    dynamic.extend(np.ones((32, 28, 28)))
    dynamic.extend(np.ones((16, 33, 9)))
    assert dynamic.shape.lower == (28, 9)
    assert dynamic.shape.upper == (33, 28)

    # fixed shape property
    fixed.extend(np.ones((9, 28, 28)))
    fixed.extend(np.ones((13, 28, 28)))
    assert fixed.shape.lower == (28, 28)
    assert fixed.shape.upper == (28, 28)
