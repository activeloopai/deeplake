import numpy as np
import pytest
from functools import reduce
import operator
from hub.api.dataset import Dataset

from hub.core.meta.dataset_meta import read_dataset_meta

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
    assert ds_new.image.shape == (4096, 4096)
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
    assert ds_new.image.shape == (4096, 4096)
    np.testing.assert_array_equal(ds_new.image.numpy(), np.ones((4, 4096, 4096)))
    ds.delete()


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


def test_compute_slices(memory_ds):
    ds = memory_ds
    shape = (64, 16, 16, 16)
    data = np.arange(reduce(operator.mul, shape, 1)).reshape(shape)
    ds.create_tensor("data")
    ds.data.extend(data)

    ss = ds.data[:].numpy()
    np.testing.assert_array_equal(ss, data[:])

    ss = ds.data[10:20].numpy()
    np.testing.assert_array_equal(ss, data[10:20])

    ss = ds.data[5].numpy()
    np.testing.assert_array_equal(ss, data[5])

    ss = ds.data[3, 3].numpy()
    np.testing.assert_array_equal(ss, data[3, 3])

    ss = ds.data[30:40, :, 8:11, 4].numpy()
    np.testing.assert_array_equal(ss, data[30:40, :, 8:11, 4])

    ss = ds.data[16, 4, 5, 1:3].numpy()
    np.testing.assert_array_equal(ss, data[16, 4, 5, 1:3])

    ss = ds.data[[0, 1, 2, 5, 6, 10, 60]].numpy()
    np.testing.assert_array_equal(ss, data[[0, 1, 2, 5, 6, 10, 60]])

    ss = ds.data[0][[0, 1, 2, 5, 6, 10, 15]].numpy()
    np.testing.assert_array_equal(ss, data[0][[0, 1, 2, 5, 6, 10, 15]])

    ss = ds.data[(0, 1, 6, 10, 15), :].numpy()
    np.testing.assert_array_equal(ss, data[(0, 1, 6, 10, 15), :])

    ss = ds.data[0][(0, 1, 6, 10, 15), :].numpy()
    np.testing.assert_array_equal(ss, data[0][(0, 1, 6, 10, 15), :])

    ss = ds.data[0, (0, 1, 5)].numpy()
    np.testing.assert_array_equal(ss, data[0, (0, 1, 5)])

    ss = ds.data[:, :][0].numpy()
    np.testing.assert_array_equal(ss, data[:, :][0])

    ss = ds.data[:, :][0:2].numpy()
    np.testing.assert_array_equal(ss, data[:, :][0:2])

    ss = ds.data[0, :][0:2].numpy()
    np.testing.assert_array_equal(ss, data[0, :][0:2])

    ss = ds.data[:, 0][0:2].numpy()
    np.testing.assert_array_equal(ss, data[:, 0][0:2])

    ss = ds.data[:, :][0][(0, 1, 2), 0].numpy()
    np.testing.assert_array_equal(ss, data[:, :][0][(0, 1, 2), 0])

    ss = ds.data[0][(0, 1, 2), 0][1].numpy()
    np.testing.assert_array_equal(ss, data[0][(0, 1, 2), 0][1])

    ss = ds.data[:, :][0][(0, 1, 2), 0][1].numpy()
    np.testing.assert_array_equal(ss, data[:, :][0][(0, 1, 2), 0][1])

    ss = ds.data[0][:].numpy()
    np.testing.assert_array_equal(ss, data[0][:])
