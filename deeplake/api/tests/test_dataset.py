import jwt

from deeplake.util.exceptions import DatasetHandlerError, UserNotLoggedInException
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


def test_allow_delete(local_ds_generator, local_path):
    ds = local_ds_generator()
    assert ds.allow_delete is True

    ds.allow_delete = False
    assert ds.allow_delete is False

    ds2 = deeplake.load(ds.path)
    assert ds2.allow_delete is False

    with pytest.raises(DatasetHandlerError):
        deeplake.empty(ds.path, overwrite=True)
        deeplake.deepcopy(src=ds.path, dest=local_path, overwrite=True)
        ds.delete()

    ds.allow_delete = True
    assert ds.allow_delete is True
    ds.delete()
