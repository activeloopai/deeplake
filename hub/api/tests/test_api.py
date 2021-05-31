import os
import shutil

import numpy as np
import pytest
from hub.api.dataset import Dataset
from hub.core.chunk_engine.read import read_dataset_meta, read_tensor_meta
from hub.core.storage import LocalProvider
from hub.core.tests.common import parametrize_all_dataset_storages


def test_persist_local(local_storage):
    if local_storage is None:
        pytest.skip()

    ds = Dataset(local_storage.root)
    ds["image"] = np.ones((4, 4096, 4096))

    ds_new = Dataset(local_storage.root)
    assert len(ds_new) == 4
    assert ds_new["image"].shape == (4096, 4096)
    np.testing.assert_array_equal(ds_new["image"].numpy(), np.ones((4, 4096, 4096)))


@parametrize_all_dataset_storages
def test_populate_dataset(ds):
    assert read_dataset_meta(ds.provider) == {"tensors": []}
    ds["image"] = np.ones((4, 28, 28))
    assert read_dataset_meta(ds.provider) == {"tensors": ["image"]}
    assert len(ds) == 4


@parametrize_all_dataset_storages
def test_compute_tensor(ds):
    ds["image"] = np.ones((32, 28, 28))
    np.testing.assert_array_equal(ds["image"].numpy(), np.ones((32, 28, 28)))


@parametrize_all_dataset_storages
def test_compute_tensor_slice(ds):
    ds["image"] = np.vstack((np.arange(16),) * 8)

    sliced_data = ds["image"][2:5].numpy()
    expected_data = np.vstack((np.arange(16),) * 3)
    np.testing.assert_array_equal(sliced_data, expected_data)


@parametrize_all_dataset_storages
def test_iterate_dataset(ds):
    labels = [1, 9, 7, 4]
    ds["image"] = np.ones((4, 28, 28))
    ds["label"] = np.asarray(labels).reshape((4, 1))

    for idx, sub_ds in enumerate(ds):
        img = sub_ds["image"].numpy()
        label = sub_ds["label"].numpy()
        np.testing.assert_array_equal(img, np.ones((1, 28, 28)))
        assert label.shape == (1, 1)
        assert label[0] == labels[idx]
