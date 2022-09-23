from deeplake.util.exceptions import DatasetHandlerError, UserNotLoggedInException
from deeplake.cli.auth import logout
from click.testing import CliRunner
import pytest
import deeplake as dl
import numpy as np


def test_new_dataset():
    with CliRunner().isolated_filesystem():
        ds = dl.dataset("test_new_dataset")
        with ds:
            ds.create_tensor("image")
            for i in range(10):
                ds.image.append(i * np.ones((100 * (i + 1), 100 * (i + 1))))

        for i in range(10):
            np.testing.assert_array_equal(
                ds.image[i].numpy(), i * np.ones((100 * (i + 1), 100 * (i + 1)))
            )


def test_dataset_empty_load():
    with CliRunner().isolated_filesystem():
        path = "test_dataset_load"

        ds = dl.empty(path)
        with ds:
            ds.create_tensor("image")
            for i in range(10):
                ds.image.append(i * np.ones((100 * (i + 1), 100 * (i + 1))))

        with pytest.raises(DatasetHandlerError):
            ds_empty = dl.empty(path)

        ds_loaded = dl.load(path)
        for i in range(10):
            np.testing.assert_array_equal(
                ds.image[i].numpy(), ds_loaded.image[i].numpy()
            )

        with pytest.raises(DatasetHandlerError):
            ds_random = dl.load("some_random_path")

        ds_overwrite_load = dl.dataset(path, overwrite=True)
        assert len(ds_overwrite_load) == 0
        assert len(ds_overwrite_load.tensors) == 0
        with ds_overwrite_load:
            ds_overwrite_load.create_tensor("image")
            for i in range(10):
                ds_overwrite_load.image.append(
                    i * np.ones((100 * (i + 1), 100 * (i + 1)))
                )

        with pytest.raises(DatasetHandlerError):
            ds_empty = dl.empty(path)

        ds_overwrite_empty = dl.dataset(path, overwrite=True)
        assert len(ds_overwrite_empty) == 0
        assert len(ds_overwrite_empty.tensors) == 0


def test_update_privacy(hub_cloud_ds):
    assert not hub_cloud_ds.public
    hub_cloud_ds.make_public()
    assert hub_cloud_ds.public
    hub_cloud_ds.make_private()
    assert not hub_cloud_ds.public

    runner = CliRunner()
    runner.invoke(logout)
    with pytest.raises(UserNotLoggedInException):
        dl.dataset(hub_cloud_ds.path)

    with pytest.raises(UserNotLoggedInException):
        dl.load(hub_cloud_ds.path)

    with pytest.raises(UserNotLoggedInException):
        dl.empty(hub_cloud_ds.path)


def test_persistence_bug(local_ds_generator):
    for tensor_name in ["abc", "abcd/defg"]:
        ds = local_ds_generator()
        with ds:
            ds.create_tensor(tensor_name)
            ds[tensor_name].append(1)

        ds = local_ds_generator()
        with ds:
            ds[tensor_name].append(2)

        ds = local_ds_generator()
        np.testing.assert_array_equal(ds[tensor_name].numpy(), np.array([[1], [2]]))
