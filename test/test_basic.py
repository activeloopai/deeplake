import pytest
import hub
import numpy as np


def test_init():
    print("- Initialize array")
    shape = (10, 10, 10, 10)
    chunk = (5, 5, 5, 5)
    datahub = hub.fs("./data/cache").connect()
    x = datahub.array(name="test/example:1", shape=shape, chunk=chunk, dtype="uint8")
    shape = np.array(shape)
    assert np.all(np.array(x.shape) == shape)
    print("passed")


def test_simple_upload_download():
    print("- Simple Chunk Upload and Download")
    datahub = hub.fs("./data/cache").connect()
    shape = (10, 10, 10, 10)
    chunk = (5, 5, 5, 5)
    datahub = hub.fs("./data/cache").connect()
    x = datahub.array(name="test/example:1", shape=shape, chunk=chunk, dtype="uint8")
    x[0] = np.ones((1, 10, 10, 10), dtype="uint8")
    assert x[0].mean() == 1
    print("passed")


def test_multiple_upload_download():
    datahub = hub.fs("./data/cache").connect()
    shape = (10, 10, 10, 10)
    chunk = (5, 5, 5, 5)
    x = datahub.array(name="test/example:1", shape=shape, chunk=chunk, dtype="uint8")
    x[0:3] = np.ones((3, 10, 10, 10), dtype="uint8")
    assert x[0:3].mean() == 1
    print("passed")


def test_cross_chunk_upload_download():
    print("- Cross Chunk Upload and Download")
    datahub = hub.fs("./data/cache").connect()
    shape = (10, 10, 10, 10)
    chunk = (5, 5, 5, 5)
    x = datahub.array(name="test/example:2", shape=shape, chunk=chunk, dtype="uint8")
    x[2:5, 0:9, 0:10] = np.ones((3, 9, 10, 10), dtype="uint8")
    x[2:5, 9:10] = np.zeros((3, 1, 10, 10), dtype="uint8")
    assert x[2:5, 0:9, 0:10].mean() == 1
    assert x[2:5, 9:10].mean() == 0
    print("passed")


def test_broadcasting():
    print("- Broadcasting")
    datahub = hub.fs("./data/cache").connect()
    shape = (100, 100, 100)
    chunk = (50, 50, 50)
    x = datahub.array(name="test/example:3", shape=shape, chunk=chunk, dtype="uint8")
    x[0, 0, 0] = 11
    assert x[0, 0, 0] == 11
    x[0] = 10
    assert x[0].mean() == 10
    x[1] = np.ones((100, 100), dtype="uint8")
    assert x[1].mean() == 1
    x[3, 90, :] = np.ones((1, 1, 100), dtype="uint8")
    assert x[3, 90].mean() == 1
    print("passed")


def test_chunk_shape():
    print("- Chunk shape")
    datahub = hub.fs("./data/cache").connect()
    shape = (10, 10, 10)
    chunk = (5, 5, 5)
    x = datahub.array(name="test/example:4", shape=shape, chunk=chunk, dtype="uint8")
    x[0:5, 0:5, 0:5] = 0
    print("passed")


def test_open_array():
    print("- Loading arrays")
    datahub = hub.fs("./data/cache").connect()
    x = datahub.open(name="test/example:4")
    print(x.shape)
    assert np.all(x.shape == np.array((10, 10, 10)))
    print("passed")


def test_squeeze_array():
    print("- Squeezing arrays")
    datahub = hub.fs("./data/cache").connect()
    shape = (10, 10, 10)
    chunk = (5, 5, 5)
    x = datahub.array(name="test/example:4", shape=shape, chunk=chunk, dtype="uint8")
    assert len(x[0].shape) == 2
    assert len(x[:1].shape) == 3
    assert len(x[0, 0, :].shape) == 1
    assert x[0, 0, 0] == 0
    print("passed")


def test_dtypes():
    print("- Numpy dtypes arrays")
    datahub = hub.fs("./data/cache").connect()
    shape = (10, 10, 10)
    chunk = (5, 5, 5)
    x = datahub.array(name="test/example:5", shape=shape, chunk=chunk, dtype="float32")
    assert x.dtype == "<f4"
    print("passed")


def test_delete_item():
    datahub = hub.fs("./data/cache").connect()
    datahub.delete(name="test/example:5")
    with pytest.raises(Exception):
        datahub.open(name="test/example:5")

