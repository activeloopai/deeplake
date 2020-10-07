import os

import numpy as np

from hub.store.storage_tensor import StorageTensor
from hub.store.store import NotZarrFolderException


def test_open():
    StorageTensor(
        "./data/test/test_storage_tensor/test_open",
        mode="w",
        shape=[50, 100, 100],
        dtype="int32",
    )

    tensor = StorageTensor("./data/test/test_storage_tensor/test_open", mode="r")
    assert tensor.shape == (50, 100, 100)
    assert tensor.chunks == (102, 204, 204)
    assert tensor.dtype == "int32"


def test_s3_open():
    StorageTensor(
        "s3://snark-test/test_storage_tensor/test_s3_open",
        mode="w",
        shape=[50, 100, 100],
        dtype="int32",
    )
    tensor = StorageTensor("s3://snark-test/test_storage_tensor/test_s3_open", mode="r")
    tensor[25:35, 10:20, 0] = np.ones((10, 10), dtype="int32")
    assert tensor.shape == (50, 100, 100)
    assert tensor.chunks == (102, 204, 204)
    assert tensor.dtype == "int32"


def test_gcs_open():
    StorageTensor(
        "gcs://snark-test/test_storage_tensor/test_gcs_open",
        mode="w",
        shape=[50, 100, 100],
        dtype="int32",
    )
    tensor = StorageTensor(
        "gcs://snark-test/test_storage_tensor/test_gcs_open", mode="r"
    )
    tensor[25:35, 10:20, 0] = np.ones((10, 10), dtype="int32")
    assert tensor.shape == (50, 100, 100)
    assert tensor.chunks == (102, 204, 204)
    assert tensor.dtype == "int32"


def test_memcache():
    tensor = StorageTensor(
        "./data/test/test_storage_tensor/test_memcache",
        mode="w",
        shape=[200, 100, 100],
        dtype="float32",
        memcache=5,
    )
    tensor[50:100, 0, 0] = np.ones((50,), dtype="float32")
    assert tensor.shape == (200, 100, 100)
    assert tensor.chunks == (256, 128, 128)
    assert tensor.dtype == "float32"


def main():
    # test_overwrite_safety()
    test_memcache()


if __name__ == "__main__":
    test_s3_open()