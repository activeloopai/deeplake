import numpy as np

from hub.core.chunk_engine import read_sample, write_array
from hub.core.chunk_engine.meta import get_meta, has_meta, validate_meta
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


def get_min_shape(batch):
    return tuple(np.minimum.reduce([sample.shape for sample in batch]))


def get_max_shape(batch):
    return tuple(np.maximum.reduce([sample.shape for sample in batch]))


def run_engine_test(arrays, storage, compression, batched, chunk_size):
    storage.clear()
    tensor_key = "tensor"

    for i, a_in in enumerate(arrays):
        write_array(
            a_in,
            tensor_key,
            compression,
            chunk_size,
            storage,
            batched=batched,
        )

        # TODO: make sure there is no more than 1 incomplete chunk at a time. because incomplete chunks are NOT compressed, if there is
        # more than 1 per tensor is inefficient

        a_out = read_sample(tensor_key, i, storage)

        # writing implicitly normalizes/batchifies shape
        a_in = normalize_and_batchify_shape(a_in, batched=batched)

        assert has_meta(tensor_key, storage), "Meta was not found."
        meta = get_meta(tensor_key, storage)
        validate_meta(
            tensor_key,
            storage,
            **{
                "compression": compression.__name__,
                "chunk_size": chunk_size,
                "length": a_in.shape[0],
                "dtype": a_in.dtype.name,
                "min_shape": get_min_shape(a_in),
                "max_shape": get_max_shape(a_in),
            },
        )

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
