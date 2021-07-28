from hub.util.exceptions import DatasetHandlerError
from click.testing import CliRunner
import pytest
import hub
import numpy as np


def test_new_dataset():
    with CliRunner().isolated_filesystem():
        ds = hub.dataset("test_new_dataset")
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

        ds = hub.empty(path)
        with ds:
            ds.create_tensor("image")
            for i in range(10):
                ds.image.append(i * np.ones((100 * (i + 1), 100 * (i + 1))))

        with pytest.raises(DatasetHandlerError):
            ds_empty = hub.empty(path)

        ds_loaded = hub.load(path)
        for i in range(10):
            np.testing.assert_array_equal(
                ds.image[i].numpy(), ds_loaded.image[i].numpy()
            )

        with pytest.raises(DatasetHandlerError):
            ds_random = hub.load("some_random_path")

        ds_overwrite_load = hub.load(path, overwrite=True)
        assert len(ds_overwrite_load) == 0
        assert len(ds_overwrite_load.tensors) == 0
        with ds_overwrite_load:
            ds_overwrite_load.create_tensor("image")
            for i in range(10):
                ds_overwrite_load.image.append(
                    i * np.ones((100 * (i + 1), 100 * (i + 1)))
                )

        with pytest.raises(DatasetHandlerError):
            ds_empty = hub.empty(path)

        ds_overwrite_empty = hub.load(path, overwrite=True)
        assert len(ds_overwrite_empty) == 0
        assert len(ds_overwrite_empty.tensors) == 0
