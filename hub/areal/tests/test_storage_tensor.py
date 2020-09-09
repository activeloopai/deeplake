import numpy as np

from hub.areal.storage_tensor import StorageTensor
from hub.areal.tensor import Tensor


def test_init():
    tensor = StorageTensor(
        "./data/test/test_storage_tensor/test_init",
        shape=[100, 100, 100],
        dtype="int32",
    )
    assert tensor.shape == (100, 100, 100)
    assert tensor.chunks == (162, 162, 162)


def test_store():
    tensor = StorageTensor(
        "./data/test/test_storage_tensor/test_store",
        shape=[200, 100, 100],
        dtype="int32",
    )
    tensor[50:100, 0, 0] = np.ones((50,), dtype="int32")
    assert tensor.shape == (200, 100, 100)
    assert tensor.chunks == (256, 128, 128)


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
