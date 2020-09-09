import os

import numpy as np

from hub.areal.storage_tensor import StorageTensor
from hub.areal.tensor import Tensor
from hub.areal.store import NotZarrFolderException


def test_open():
    StorageTensor(
        "./data/test/test_storage_tensor/test_open",
        shape=[50, 100, 100],
        dtype="int32",
    )

    tensor = StorageTensor(
        "./data/test/test_storage_tensor/test_open",
    )
    assert tensor.shape == (50, 100, 100)
    assert tensor.chunks == (102, 204, 204)
    assert tensor.dtype == "int32"


def test_s3_open():
    StorageTensor(
        "s3://snark-test/test_storage_tensor/test_s3_open",
        shape=[50, 100, 100],
        dtype="int32",
    )
    tensor = StorageTensor("s3://snark-test/test_storage_tensor/test_s3_open")
    tensor[25:35, 10:20, 0] = np.ones((10, 10), dtype="int32")
    assert tensor.shape == (50, 100, 100)
    assert tensor.chunks == (102, 204, 204)
    assert tensor.dtype == "int32"


def test_gcs_open():
    StorageTensor(
        "gcs://snark-test/test_storage_tensor/test_gcs_open",
        shape=[50, 100, 100],
        dtype="int32",
    )
    tensor = StorageTensor("gcs://snark-test/test_storage_tensor/test_gcs_open")
    tensor[25:35, 10:20, 0] = np.ones((10, 10), dtype="int32")
    assert tensor.shape == (50, 100, 100)
    assert tensor.chunks == (102, 204, 204)
    assert tensor.dtype == "int32"


def test_memcache():
    tensor = StorageTensor(
        "./data/test/test_storage_tensor/test_memcache",
        shape=[200, 100, 100],
        dtype="float32",
        memcache=5,
    )
    tensor[50:100, 0, 0] = np.ones((50,), dtype="float32")
    assert tensor.shape == (200, 100, 100)
    assert tensor.chunks == (256, 128, 128)
    assert tensor.dtype == "float32"


def test_overwrite_safety():
    path = "./data/test/test_storage_tensor/test_overwrite_safety"
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "hello.txt"), "w") as f:
        f.write("Hello World")
    try:
        StorageTensor(
            "./data/test/test_storage_tensor/test_overwrite_safety",
            shape=[200, 100, 100],
        )
    except Exception as ex:
        assert isinstance(ex, NotZarrFolderException)
    else:
        assert False, "Should have raised Exception didn't"


def test_overwrite_safety_folder():
    path = "./data/test/test_storage_tensor/test_overwrite_safety_folder"
    os.makedirs(os.path.join(path, "inner_folder"), exist_ok=True)
    try:
        StorageTensor(
            "./data/test/test_storage_tensor/test_overwrite_safety_folder",
            shape=[200, 100, 100],
        )
    except Exception as ex:
        assert isinstance(ex, NotZarrFolderException)
    else:
        assert False, "Should have raised Exception didn't"


def main():
    test_overwrite_safety()


if __name__ == "__main__":
    main()