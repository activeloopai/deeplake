import jwt

from deeplake.util.exceptions import DatasetHandlerError, UserNotLoggedInException
from deeplake.cli.auth import login, logout
from click.testing import CliRunner
import pytest
import deeplake
import numpy as np


def test_new_dataset():
    with CliRunner().isolated_filesystem():
        ds = deeplake.dataset("test_new_dataset")
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

        ds = deeplake.empty(path)
        with ds:
            ds.create_tensor("image")
            for i in range(10):
                ds.image.append(i * np.ones((100 * (i + 1), 100 * (i + 1))))

        with pytest.raises(DatasetHandlerError):
            ds_empty = deeplake.empty(path)

        ds_loaded = deeplake.load(path)
        for i in range(10):
            np.testing.assert_array_equal(
                ds.image[i].numpy(), ds_loaded.image[i].numpy()
            )

        with pytest.raises(DatasetHandlerError):
            ds_random = deeplake.load("some_random_path")

        ds_overwrite_load = deeplake.dataset(path, overwrite=True)
        assert len(ds_overwrite_load) == 0
        assert len(ds_overwrite_load.tensors) == 0
        with ds_overwrite_load:
            ds_overwrite_load.create_tensor("image")
            for i in range(10):
                ds_overwrite_load.image.append(
                    i * np.ones((100 * (i + 1), 100 * (i + 1)))
                )

        with pytest.raises(DatasetHandlerError):
            ds_empty = deeplake.empty(path)

        ds_overwrite_empty = deeplake.dataset(path, overwrite=True)
        assert len(ds_overwrite_empty) == 0
        assert len(ds_overwrite_empty.tensors) == 0


@pytest.mark.slow
def test_update_privacy(hub_cloud_ds):
    assert not hub_cloud_ds.public
    hub_cloud_ds.make_public()
    assert hub_cloud_ds.public
    hub_cloud_ds.make_private()
    assert not hub_cloud_ds.public

    runner = CliRunner()
    runner.invoke(logout)
    with pytest.raises(UserNotLoggedInException):
        deeplake.dataset(hub_cloud_ds.path)

    with pytest.raises(UserNotLoggedInException):
        deeplake.load(hub_cloud_ds.path)

    with pytest.raises(UserNotLoggedInException):
        deeplake.empty(hub_cloud_ds.path)


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


def test_dataset_token(local_ds_generator, hub_cloud_dev_credentials):
    username, password = hub_cloud_dev_credentials
    CliRunner().invoke(login, f"-u {username} -p {password}")
    ds = local_ds_generator()
    token = ds.token
    token_username = jwt.decode(token, options={"verify_signature": False})["id"]
    assert token_username == username

    CliRunner().invoke(logout)
    ds = local_ds_generator()
    assert ds.token is None
