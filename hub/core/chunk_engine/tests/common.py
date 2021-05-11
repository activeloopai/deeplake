import numpy as np
import pickle

from hub.core.chunk_engine import write_array, read_array
from hub.core.chunk_engine.util import (
    normalize_and_batchify_shape,
    get_meta_key,
    get_index_map_key,
    get_chunk_key,
)
from hub.core.storage import MemoryProvider
from hub.core.typing import Provider

from typing import List, Tuple


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


def get_min_shape(batch: np.ndarray) -> Tuple:
    return tuple(np.minimum.reduce([sample.shape for sample in batch]))


def get_max_shape(batch: np.ndarray) -> Tuple:
    return tuple(np.maximum.reduce([sample.shape for sample in batch]))


def assert_incomplete_chunk_count_is_valid(index_map: List):
    incomplete_count = 0
    for entry in index_map:
        if "incomplete_chunk_names" in entry:
            incomplete_count += len(entry["incomplete_chunk_names"])
    assert incomplete_count <= 1, (
        "Number of incomplete chunks should never exceed 1. Got: %i" % incomplete_count
    )


def assert_chunk_sizes(key: str, index_map: List, chunk_size: int, storage: Provider):
    for i, entry in enumerate(index_map):
        for j, chunk_name in enumerate(entry["chunk_names"]):
            chunk_key = get_chunk_key(key, chunk_name)
            chunk_length = len(storage[chunk_key])
            assert (
                chunk_length == chunk_size
            ), 'Chunk "%s" didn\'t match chunk_size=%i (got %i) @ [%i, %i].' % (
                chunk_name,
                chunk_size,
                chunk_length,
                i,
                j,
            )


def run_engine_test(arrays, storage, batched, chunk_size):
    storage.clear()
    tensor_key = "tensor"

    for i, a_in in enumerate(arrays):
        write_array(
            a_in,
            tensor_key,
            chunk_size,
            storage,
            batched=batched,
        )

        index_map_key = get_index_map_key(tensor_key)
        index_map = pickle.loads(storage[index_map_key])

        assert_incomplete_chunk_count_is_valid(index_map)
        assert_chunk_sizes(tensor_key, index_map, chunk_size, storage)

        # `write_array` implicitly normalizes/batchifies shape
        a_in = normalize_and_batchify_shape(a_in, batched=batched)

        a_out = read_array(tensor_key, storage)

        meta_key = get_meta_key(tensor_key)
        assert meta_key in storage, "Meta was not found."
        meta = pickle.loads(storage[meta_key])

        assert_meta_is_valid(
            meta,
            {
                "chunk_size": chunk_size,
                "length": a_in.shape[0],
                "dtype": a_in.dtype.name,
                "min_shape": get_min_shape(a_in),
                "max_shape": get_max_shape(a_in),
            },
        )

        assert np.array_equal(a_in, a_out), "Array not equal @ batch_index=%i." % i

    storage.clear()


def get_random_array(shape: Tuple, dtype: str) -> np.ndarray:
    if "int" in dtype:
        low = np.iinfo(dtype).min
        high = np.iinfo(dtype).max
        return np.random.randint(low=low, high=high, size=shape, dtype=dtype)

    if "float" in dtype:
        return np.random.uniform(shape).astype(dtype)

    if "bool" in dtype:
        a = np.random.uniform(shape)
        return a > 0.5

    assert False, 'Invalid dtype "%s".' % dtype


def assert_meta_is_valid(meta: dict, expected_meta: dict):
    for k, v in expected_meta.items():
        assert k in meta
        assert v == meta[k]
