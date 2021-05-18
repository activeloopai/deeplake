import os, shutil
import numpy as np
from hub.api.dataset import Dataset
from hub.core.storage import LocalProvider
from hub.core.chunk_engine.read import read_dataset_meta, read_tensor_meta
import pytest


@pytest.fixture(autouse=True)
def clear_test_files():
    shutil.rmtree("/tmp/hub-test/", ignore_errors=True)


def test_create_empty_memory_dataset():
    ds = Dataset("username/data")
    assert ds.slice == slice(None)
    assert read_dataset_meta(ds.provider) == {"tensors": []}


def test_create_empty_local_dataset():
    ds = Dataset("/tmp/hub-test/empty")
    assert type(ds.provider) == LocalProvider
    assert read_dataset_meta(ds.provider) == {"tensors": []}


def test_populate_local_dataset():
    ds = Dataset("/tmp/hub-test/populate")
    assert read_dataset_meta(ds.provider) == {"tensors": []}
    ds["image"] = np.ones((4, 28, 28))
    assert read_dataset_meta(ds.provider) == {"tensors": ["image"]}
    assert len(ds) == 4


def test_persist_local_meta():
    ds = Dataset("/tmp/hub-test/persist-meta")
    ds["image"] = np.ones((4, 28, 28))

    ds_new = Dataset("/tmp/hub-test/persist-meta")
    assert len(ds_new) == 4
    assert ds_new["image"].shape == (28, 28)


def test_compute_local_tensor():
    ds = Dataset("/tmp/hub-test/compute")
    ds["image"] = np.ones((32, 28, 28))
    np.testing.assert_array_equal(ds["image"].numpy(), np.ones((32, 28, 28)))


def test_persist_compute_local_tensor():
    ds = Dataset("/tmp/hub-test/persist-compute")
    ds["image"] = np.ones((4, 4096, 4096))

    ds_new = Dataset("/tmp/hub-test/persist-compute")
    np.testing.assert_array_equal(ds["image"].numpy(), np.ones((4, 4096, 4096)))


def test_compute_tensor_slice():
    ds = Dataset("/tmp/hub-test/compute-slice")
    ds["image"] = np.vstack((np.arange(16),) * 8)

    sliced_data = ds["image"][2:5].numpy()
    expected_data = np.vstack((np.arange(16),) * 3)
    np.testing.assert_array_equal(sliced_data, expected_data)


def test_iterate_dataset():
    labels = [1, 9, 7, 4]
    ds = Dataset("/tmp/hub-test/iterate")
    ds["image"] = np.ones((4, 28, 28))
    ds["label"] = np.asarray(labels).reshape((4, 1))

    for idx, sub_ds in enumerate(ds):
        img = sub_ds["image"].numpy()
        label = sub_ds["label"].numpy()
        np.testing.assert_array_equal(img, np.ones((1, 28, 28)))
        assert label.shape == (1, 1)
        assert label[0] == labels[idx]
