import pytest
import numpy as np
import pickle

from hub.core.chunk_engine import write_array, read_array
from hub.core.chunk_engine.util import normalize_and_batchify_shape

from hub.util.s3 import has_s3_credentials
from hub.util.keys import get_meta_key, get_index_map_key, get_chunk_key
from hub.core.storage import MappedProvider, S3Provider
from hub.core.typing import Provider

from typing import List, Tuple

from uuid import uuid1


def random_key(prefix="test_"):
    return prefix + str(uuid1())


STORAGE_PROVIDERS = (
    MappedProvider(),
    S3Provider("snark-test/hub-2.0/core/common/%s" % random_key("session_")),
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


def run_engine_test(
    arrays: List[np.ndarray], storage: Provider, batched: bool, chunk_size: int
):
    clear_if_memory_provider(storage)

    key = random_key()

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

    clear_if_memory_provider(storage)


def benchmark_write(
    key, arrays, chunk_size, storage, batched, clear_memory_after_write=True
):
    clear_if_memory_provider(storage)

    for a_in in arrays:
        write_array(
            a_in,
            key,
            chunk_size,
            storage,
            batched=batched,
        )

    if clear_memory_after_write:
        clear_if_memory_provider(storage)


def benchmark_read(key: str, storage: Provider):
    read_array(key, storage)


def skip_if_no_required_creds(storage: Provider):
    """If `storage` is a provider that requires creds, and they are not found, skip the current test."""

    if type(storage) == S3Provider:
        if not has_s3_credentials():
            pytest.skip()


def clear_if_memory_provider(storage: Provider):
    """If `storage` is memory-based, clear it."""

    if type(storage) == MappedProvider:
        storage.clear()
