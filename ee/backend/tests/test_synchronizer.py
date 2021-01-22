from os import sync
import pytest
import cloudpickle
import fsspec
import posixpath
import numpy as np
import zarr

from pathos.pools import ProcessPool
from ee.backend.utils import redis_loaded
from ee.backend.synchronizer import RedisSynchronizer

import hub
from hub.schema import Tensor, Image, Text
from hub.utils import Timer
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


@pytest.mark.skipif(
    not redis_loaded(),
    reason="requires redis to be loaded",
)
def test_synchronization():
    t = DynamicTensor(
        create_store("./data/test/test_dynamic_tensor"),
        mode="w",
        shape=(5, 100, 100),
        max_shape=(5, 100, 100),
        dtype="int32",
        synchronizer=RedisSynchronizer(using_ray=False),
    )
    pool = ProcessPool(nodes=4)

    samples = [1, 2, 3, 4]

    def store(index):
        t[index, :, :] = index * np.ones((100, 100), dtype="int32")

    pool.map(store, samples)

    for i in samples:
        assert t[i, 0, :10].tolist() == [i] * 10


my_schema = {
    "image": Tensor((28, 28, 4), "int32", (28, 28, 4)),
    "label": Text((None,), "int64", (20,)),
    "confidence": "float",
}


dynamic_schema = {
    "image": Tensor(shape=(None, None, None), dtype="int32", max_shape=(32, 32, 3)),
    "label": Text((None,), "int64", (20,)),
}


@pytest.mark.skipif(
    not redis_loaded(),
    reason="requires redis to be loaded",
)
def test_pipeline():
    synchronizer = RedisSynchronizer(using_ray=False)
    ds = hub.Dataset(
        "./data/test/test_pipeline_multiple2",
        mode="w",
        shape=(100,),
        schema=my_schema,
        synchronizer=synchronizer,
    )

    with Timer("multiple pipes"):

        @hub.transform(
            schema=my_schema,
            scheduler="processed",
            synchronizer=synchronizer,
            workers=2,
        )
        def my_transform(sample, multiplier: int = 2):
            if not isinstance(sample, int):
                sample = sample["image"]
            else:
                sample = sample * np.ones((28, 28, 4), dtype="int32")

            return {
                "image": sample * multiplier,
                "label": f"hello",
                "confidence": 0.2 * multiplier,
            }

        out_ds = my_transform(range(len(ds)), multiplier=2)
        out_ds = my_transform(out_ds, multiplier=2)
        out_ds = out_ds.store("./data/test/test_pipeline_multiple_5")

        for el in range(len(ds)):
            assert (out_ds["image", el].compute() == el * 4).all()


@pytest.mark.skipif(
    not redis_loaded(),
    reason="requires redis to be loaded",
)
def test_counter():

    a = RedisSynchronizer(using_ray=False)
    a.reset(key="key1")
    a.append(key="key1", number=10)
    assert a.get(key="key1") == 10

    pool = ProcessPool(nodes=4)
    samples = [1, 2, 3, 4]

    def store(index):
        RedisSynchronizer().append(key="key1", number=10)

    pool.map(store, samples)
    RedisSynchronizer(using_ray=False).append(key="key1", number=-10)
    assert a.get(key="key1") == 40


if __name__ == "__main__":
    test_synchronization()
    test_pipeline()