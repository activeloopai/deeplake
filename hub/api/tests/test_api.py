import numpy as np
import pytest
from hub.api.dataset import Dataset

from hub.core.meta.dataset_meta import read_dataset_meta

from hub.core.tests.common import parametrize_all_dataset_storages


def test_persist_local(local_storage):
    if local_storage is None:
        pytest.skip()

    ds = Dataset(local_storage.root)
    ds.create_tensor("image")
    ds.image.extend(np.ones((4, 4096, 4096)))

    ds_new = Dataset(local_storage.root)
    assert len(ds_new) == 4
    assert ds_new.image.shape == (4096, 4096)
    np.testing.assert_array_equal(ds_new.image.numpy(), np.ones((4, 4096, 4096)))


@parametrize_all_dataset_storages
def test_populate_dataset(ds):
    assert ds.meta == {"tensors": []}
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

    assert ds.meta == {"tensors": ["image"]}


@parametrize_all_dataset_storages
def test_compute_tensor(ds):
    ds.create_tensor("image")
    ds.image.extend(np.ones((32, 28, 28)))
    np.testing.assert_array_equal(ds.image.numpy(), np.ones((32, 28, 28)))


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
