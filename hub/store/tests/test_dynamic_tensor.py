from hub.exceptions import DynamicTensorShapeException
import posixpath

import numpy as np
import fsspec
from zarr.creation import create

from hub.store.dynamic_tensor import DynamicTensor
from hub.store.store import StorageMapWrapperWithCommit


def create_store(path: str, overwrite=True):
    fs: fsspec.AbstractFileSystem = fsspec.filesystem("file")
    if fs.exists(path) and overwrite:
        fs.rm(path, recursive=True)
    dynpath = posixpath.join(path, "--dynamic--")
    if not fs.exists(dynpath):
        fs.makedirs(dynpath)
    mapper = fs.get_mapper(path)
    mapper["--dynamic--/hello.txt"] = bytes("Hello World", "utf-8")
    return StorageMapWrapperWithCommit(mapper)


def test_read_and_append_modes():
    t = DynamicTensor(
        create_store("./data/test/test_read_and_append_modes"),
        mode="a",
        shape=(5, 100, 100),
        max_shape=(5, 100, 100),
        dtype="int32",
    )
    t[0, 80:, 80:] = np.ones((20, 20), dtype="int32")
    assert t[0, -5, 90:].tolist() == [1] * 10
    t.flush()
    t.close()

    t = DynamicTensor(
        create_store("./data/test/test_read_and_append_modes", overwrite=False),
        mode="r",
    )
    t.get_shape(0) == (100, 100)
    assert t[0, -5, 90:].tolist() == [1] * 10
    t.close()


def test_dynamic_tensor():
    t = DynamicTensor(
        create_store("./data/test/test_dynamic_tensor"),
        mode="w",
        shape=(5, 100, 100),
        max_shape=(5, 100, 100),
        dtype="int32",
    )
    t[0, 80:, 80:] = np.ones((20, 20), dtype="int32")
    assert t[0, -5, 90:].tolist() == [1] * 10


def test_dynamic_tensor_shape_none():
    try:
        DynamicTensor(
            create_store("./data/test/test_dynamic_tensor_shape_none"),
            mode="w",
            dtype="int32",
        )
    except TypeError as ex:
        assert "shape cannot be none" in str(ex)


def test_dynamic_tensor_2():
    t = DynamicTensor(
        create_store("./data/test/test_dynamic_tensor_2"),
        mode="w",
        shape=(5, None, None),
        max_shape=(5, 100, 100),
        dtype="int32",
    )
    t[0] = np.ones((10, 10), dtype="int32")
    assert t[0, 5].tolist() == [1] * 10
    assert t[0, 5, :].tolist() == [1] * 10
    t[0, 6] = 2 * np.ones((20,), dtype="int32")
    assert t[0, 5, :].tolist() == [1] * 10 + [0] * 10
    assert t.get_shape(0).tolist() == [10, 20]
    assert t.get_shape(slice(0, 1)).tolist() == [1, 10, 20]


def test_dynamic_tensor_3():
    t = DynamicTensor(
        create_store("./data/test/test_dynamic_tensor_3"),
        mode="w",
        shape=(5, None, None, None),
        max_shape=(5, 100, 100, 100),
        dtype="int32",
    )
    t[0, 5] = np.ones((20, 30), dtype="int32")
    t[0, 6:8, 5:9] = 5 * np.ones((2, 4, 30), dtype="int32")
    assert t[0, 5, 7].tolist() == [1] * 30
    assert t[0, 7, 8].tolist() == [5] * 30


def test_dynamic_tensor_shapes():
    t = DynamicTensor(
        create_store("./data/test/test_dynamic_tensor_5"),
        mode="w",
        shape=(5, None, None),
        max_shape=(5, 100, 100),
        dtype="int32",
    )
    t[0] = np.ones((5, 10), dtype="int32")
    t[0, 6] = 2 * np.ones((20,), dtype="int32")
    assert t[0, -1].tolist() == [2] * 20
    t.close()


def test_dynamic_tensor_4():
    t = DynamicTensor(
        create_store("./data/test/test_dynamic_tensor_6"),
        mode="w",
        shape=(5, None, None, None),
        max_shape=(5, 100, 100, 10),
        dtype="int32",
    )
    t[0, 6:8] = np.ones((2, 20, 10), dtype="int32")
    assert (t[0, 6:8] == np.ones((2, 20, 10), dtype="int32")).all()


if __name__ == "__main__":
    test_read_and_append_modes()
    # test_chunk_iterator()
    # test_dynamic_tensor_shapes()
