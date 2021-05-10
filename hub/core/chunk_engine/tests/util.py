import numpy as np

from hub.core.chunk_engine import read_sample, write_array
from hub.core.chunk_engine.util import normalize_and_batchify_shape
from hub.core.storage import MemoryProvider

from hub.core.chunk_engine.dummy_util import (
    DummySampleCompression,
    DummyChunkCompression,
)


ROOT = "PYTEST_TENSOR_COLLECTION"
STORAGE_PROVIDERS = (MemoryProvider(ROOT),)


CHUNK_SIZES = (
    128,
    4096,
    16000000,
)


DTYPES = (
    "uint8",
    "int64",
    "float64",
    "bool",
)


COMPRESSIONS = (
    DummySampleCompression(),
    DummyChunkCompression(),
)


def run_engine_test(arrays, storage, compression, batched, chunk_size):
    storage.clear()

    for i, a_in in enumerate(arrays):
        write_array(
            a_in,
            "tensor",
            compression,
            chunk_size,
            storage,
            batched=batched,
        )

        # TODO: make sure there is no more than 1 incomplete chunk at a time. because incomplete chunks are NOT compressed, if there is
        # more than 1 per tensor it can get inefficient

        a_out = read_sample("tensor", i, storage)

        # writing implicitly normalizes/batchifies shape
        a_in = normalize_and_batchify_shape(a_in, batched=batched)

        print("in:")
        print(a_in)
        print("out:")
        print(a_out)

        assert np.array_equal(a_in, a_out), "Array not equal @ batch_index=%i." % i

    storage.clear()


def get_random_array(shape, dtype):
    if "int" in dtype:
        low = np.iinfo(dtype).min
        high = np.iinfo(dtype).max
        return np.random.randint(low=low, high=high, size=shape, dtype=dtype)

    if "float" in dtype:
        return np.random.uniform(shape).astype(dtype)

    if "bool" in dtype:
        a = np.random.uniform(shape)
        return a > 0.5
