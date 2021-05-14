import os
import pytest
import numpy as np
import pickle
from uuid import uuid1

from hub.core.chunk_engine import write_array, read_array
from hub.core.storage import MemoryProvider, S3Provider

from hub.util.array import normalize_and_batchify_shape
from hub.util.s3 import has_s3_credentials
from hub.util.keys import get_meta_key, get_index_map_key, get_chunk_key

from hub.core.tests.common import current_test_name  # type: ignore
from typing import List, Tuple
from hub.core.typing import StorageProvider

from uuid import uuid1


def random_key(prefix="test_"):
    return prefix + str(uuid1())


STORAGE_PROVIDERS = (
    MemoryProvider("snark-test/hub-2.0/core/common/%s" % random_key("session_")),
    S3Provider("snark-test/hub-2.0/core/common/%s" % random_key("session_")),
)


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


def get_random_array(shape: Tuple[int], dtype: str) -> np.ndarray:
    dtype = dtype.lower()

    if "int" in dtype:
        low = np.iinfo(dtype).min
        high = np.iinfo(dtype).max
        return np.random.randint(low=low, high=high, size=shape, dtype=dtype)

    if "float" in dtype:
        # get float16 because np.random.uniform doesn't support the `dtype` argument.
        low = np.finfo("float16").min
        high = np.finfo("float16").max
        return np.random.uniform(low=low, high=high, size=shape).astype(dtype)

    if "bool" in dtype:
        a = np.random.uniform(size=shape)
        return a > 0.5

    raise ValueError("Dtype %s not supported." % dtype)


def assert_meta_is_valid(meta: dict, expected_meta: dict):
    for k, v in expected_meta.items():
        assert k in meta
        assert v == meta[k]


def assert_chunk_sizes(
    key: str, index_map: List, chunk_size: int, storage: StorageProvider
):
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


def run_engine_test(
    arrays: List[np.ndarray], storage: StorageProvider, batched: bool, chunk_size: int
):
    key = current_test_name(with_uuid=True)

    for i, a_in in enumerate(arrays):
        write_array(
            a_in,
            key,
            chunk_size,
            storage,
            batched=batched,
        )

        index_map_key = get_index_map_key(key)
        index_map = pickle.loads(storage[index_map_key])

        assert_chunk_sizes(key, index_map, chunk_size, storage)

        # `write_array` implicitly normalizes/batchifies shape
        a_in = normalize_and_batchify_shape(a_in, batched=batched)

        a_out = read_array(key, storage)

        meta_key = get_meta_key(key)
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


def clear_unless_s3(storage):
    """Clears `storage` unless it is an S3Provider. Returns True if cleared, False otherwise."""

    if type(storage) != S3Provider:
        # don't clear S3Provider (this is very slow)
        storage.clear()
        return True
    return False


def benchmark_write(key, arrays, chunk_size, storage, batched):
    """Benchmarks `write_array`. Clears storage (unless it is an S3Provider) before writing arrays.

    Returns:
        str: The key that `write_array` was called with. If `storage` is an `S3Provider`, this key will contain
            a random uuid as it's subpath.
    """

    actual_key = key
    if not clear_unless_s3(storage):
        actual_key = os.path.join(key, str(uuid1()))

    for a_in in arrays:
        write_array(
            a_in,
            actual_key,
            chunk_size,
            storage,
            batched=batched,
        )

    return key


def benchmark_read(key: str, storage: StorageProvider):
    read_array(key, storage)
