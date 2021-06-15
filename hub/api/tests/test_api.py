import os
import numpy as np
import pytest

import hub
from hub.api.dataset import Dataset
from hub.core.tests.common import parametrize_all_dataset_storages
from hub.client.utils import has_hub_testing_creds, write_token
from hub.client.client import HubBackendClient


def test_persist_local(local_storage):
    if local_storage is None:
        pytest.skip()

    ds = Dataset(local_storage.root, local_cache_size=512)
    ds.create_tensor("image")
    ds.image.extend(np.ones((4, 4096, 4096)))

    ds_new = Dataset(local_storage.root)
    assert len(ds_new) == 4

    assert ds_new.image.shape.lower == (4096, 4096)
    assert ds_new.image.shape.upper == (4096, 4096)

    np.testing.assert_array_equal(ds_new.image.numpy(), np.ones((4, 4096, 4096)))
    ds.delete()


def test_persist_with_local(local_storage):
    if local_storage is None:
        pytest.skip()

    with Dataset(local_storage.root, local_cache_size=512) as ds:

        ds.create_tensor("image")
        ds.image.extend(np.ones((4, 4096, 4096)))

        ds_new = Dataset(local_storage.root)
        assert len(ds_new) == 0  # shouldn't be flushed yet

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
    assert ds.meta == {"tensors": [], "version": hub.__version__}
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

    assert ds.meta == {"tensors": ["image"], "version": hub.__version__}


def test_stringify(memory_ds):
    ds = memory_ds
    ds.create_tensor("image")
    ds.image.extend(np.ones((4, 4)))
    assert (
        str(ds) == "Dataset(path=hub_pytest/test_api/test_stringify, tensors=['image'])"
    )
    assert (
        str(ds[1:2])
        == "Dataset(path=hub_pytest/test_api/test_stringify, index=Index([slice(1, 2, 1)]), tensors=['image'])"
    )
    assert str(ds.image) == "Tensor(key='image')"
    assert str(ds[1:2].image) == "Tensor(key='image', index=Index([slice(1, 2, 1)]))"


def test_stringify_with_path(local_ds):
    ds = local_ds
    assert local_ds.path
    assert str(ds) == f"Dataset(path={local_ds.path}, tensors=[])"


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


def _check_tensor(tensor, data):
    np.testing.assert_array_equal(tensor.numpy(), data)


def test_compute_slices(memory_ds):
    ds = memory_ds
    shape = (64, 16, 16, 16)
    data = np.arange(np.prod(shape)).reshape(shape)
    ds.create_tensor("data")
    ds.data.extend(data)

    _check_tensor(ds.data[:], data[:])
    _check_tensor(ds.data[10:20], data[10:20])
    _check_tensor(ds.data[5], data[5])
    _check_tensor(ds.data[0][:], data[0][:])
    _check_tensor(ds.data[3, 3], data[3, 3])
    _check_tensor(ds.data[30:40, :, 8:11, 4], data[30:40, :, 8:11, 4])
    _check_tensor(ds.data[16, 4, 5, 1:3], data[16, 4, 5, 1:3])
    _check_tensor(ds[[0, 1, 2, 5, 6, 10, 60]].data, data[[0, 1, 2, 5, 6, 10, 60]])
    _check_tensor(ds.data[[0, 1, 2, 5, 6, 10, 60]], data[[0, 1, 2, 5, 6, 10, 60]])
    _check_tensor(ds.data[0][[0, 1, 2, 5, 6, 10, 15]], data[0][[0, 1, 2, 5, 6, 10, 15]])
    _check_tensor(ds[(0, 1, 6, 10, 15), :].data, data[(0, 1, 6, 10, 15), :])
    _check_tensor(ds.data[(0, 1, 6, 10, 15), :], data[(0, 1, 6, 10, 15), :])
    _check_tensor(ds.data[0][(0, 1, 6, 10, 15), :], data[0][(0, 1, 6, 10, 15), :])
    _check_tensor(ds.data[0, (0, 1, 5)], data[0, (0, 1, 5)])
    _check_tensor(ds.data[:, :][0], data[:, :][0])
    _check_tensor(ds.data[:, :][0:2], data[:, :][0:2])
    _check_tensor(ds.data[0, :][0:2], data[0, :][0:2])
    _check_tensor(ds.data[:, 0][0:2], data[:, 0][0:2])
    _check_tensor(ds.data[:, 0][0:2], data[:, 0][0:2])
    _check_tensor(ds.data[:, :][0][(0, 1, 2), 0], data[:, :][0][(0, 1, 2), 0])
    _check_tensor(ds.data[0][(0, 1, 2), 0][1], data[0][(0, 1, 2), 0][1])
    _check_tensor(ds.data[:, :][0][(0, 1, 2), 0][1], data[:, :][0][(0, 1, 2), 0][1])


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


@pytest.mark.skipif(not has_hub_testing_creds(), reason="requires hub credentials")
def test_hub_cloud_dataset():
    username = "testingacc"
    password = os.getenv("ACTIVELOOP_HUB_PASSWORD")
    client = HubBackendClient()
    token = client.request_auth_token(username, password)
    write_token(token)
    ds = Dataset("hub://testingacc/hub2ds")
    for i in range(10):
        np.testing.assert_array_equal(ds.image[i].numpy(), i * np.ones((100, 100)))
