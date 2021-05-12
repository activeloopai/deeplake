import numpy as np
import pickle

from hub.core.chunk_engine import write_array, read_array
from hub.core.chunk_engine.util import (
    normalize_and_batchify_shape,
    get_meta_key,
    get_index_map_key,
    get_chunk_key,
    get_random_array,
)
from hub.core.storage import MappedProvider
from hub.core.typing import Provider

from typing import List, Tuple


TENSOR_KEY = "TEST_TENSOR"


STORAGE_PROVIDERS = (
    MappedProvider(),
)  # TODO: replace MappedProvider with MemoryProvider


CHUNK_SIZES = (
    128,
    4096,
    16000000,  # 16MB
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


def assert_meta_is_valid(meta: dict, expected_meta: dict):
    for k, v in expected_meta.items():
        assert k in meta
        assert v == meta[k]


def assert_chunk_sizes(key: str, index_map: List, chunk_size: int, storage: Provider):
    incomplete_chunk_names = set()
    complete_chunk_count = 0
    total_chunks = 0
    for i, entry in enumerate(index_map):
        for j, chunk_name in enumerate(entry["chunk_names"]):
            chunk_key = get_chunk_key(key, chunk_name)
            chunk_length = len(storage[chunk_key])

            # exceeding chunk_size is never acceptable
            assert (
                chunk_length <= chunk_size
            ), 'Chunk "%s" exceeded chunk_size=%i (got %i) @ [%i, %i].' % (
                chunk_name,
                chunk_size,
                chunk_length,
                i,
                j,
            )

            if chunk_length < chunk_size:
                incomplete_chunk_names.add(chunk_name)
            if chunk_length == chunk_size:
                complete_chunk_count += 1

            total_chunks += 1

    incomplete_chunk_count = len(incomplete_chunk_names)
    assert (
        incomplete_chunk_count <= 1
    ), "Incomplete chunk count should never exceed 1. Incomplete count: %i. Complete count: %i. Total: %i.\nIncomplete chunk names: %s" % (
        incomplete_chunk_count,
        complete_chunk_count,
        total_chunks,
        str(incomplete_chunk_names),
    )


def run_engine_test(arrays, storage, batched, chunk_size):
    storage.clear()

    for i, a_in in enumerate(arrays):
        write_array(
            a_in,
            TENSOR_KEY,
            chunk_size,
            storage,
            batched=batched,
        )

        index_map_key = get_index_map_key(TENSOR_KEY)
        index_map = pickle.loads(storage[index_map_key])

        assert_chunk_sizes(TENSOR_KEY, index_map, chunk_size, storage)

        # `write_array` implicitly normalizes/batchifies shape
        a_in = normalize_and_batchify_shape(a_in, batched=batched)

        a_out = read_array(TENSOR_KEY, storage)

        meta_key = get_meta_key(TENSOR_KEY)
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


def benchmark_write(arrays, chunk_size, storage, batched, clear_after_write=True):
    storage.clear()

    for a_in in arrays:
        write_array(
            a_in,
            TENSOR_KEY,
            chunk_size,
            storage,
            batched=batched,
        )

    if clear_after_write:
        storage.clear()


def benchmark_read(storage):
    read_array(TENSOR_KEY, storage)
