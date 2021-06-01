from typing import Dict, List, Tuple

import numpy as np

from hub.core.tensor import (
    add_samples_to_tensor,
    create_tensor,
    tensor_exists,
    read_samples_from_tensor,
)
from hub.core.meta.tensor_meta import read_tensor_meta, tensor_meta_from_array
from hub.core.meta.index_map import read_index_map

from hub.core.typing import StorageProvider
from hub.tests.common import TENSOR_KEY
from hub.util.array import normalize_and_batchify_shape
from hub.util.keys import get_chunk_key


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
        assert k in meta, 'Key "%s" not found in meta: %s' % (k, str(meta))
        assert (
            v == meta[k]
        ), 'Value for key "%s" mismatch.\n(actual): %s\n!=\n(expected):%s' % (
            k,
            meta[k],
            v,
        )


def assert_chunk_sizes(
    key: str, index_map: List, chunk_size: int, storage: StorageProvider
):
    incomplete_chunk_names = set()
    complete_chunk_count = 0
    total_chunks = 0
    actual_chunk_lengths_dict: Dict[str, int] = {}
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

            if chunk_name in actual_chunk_lengths_dict:
                assert (
                    chunk_length == actual_chunk_lengths_dict[chunk_name]
                ), "Chunk size changed from one read to another."
            else:
                actual_chunk_lengths_dict[chunk_name] = chunk_length

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

    # assert that all chunks (except the last one) are of expected size (`chunk_size`)
    actual_chunk_lengths = np.array(list(actual_chunk_lengths_dict.values()))
    if len(actual_chunk_lengths) > 1:
        candidate_chunk_lengths = actual_chunk_lengths[:-1]
        assert np.all(
            candidate_chunk_lengths == chunk_size
        ), "All chunks (except the last one) MUST be == `chunk_size`. chunk_size=%i\n\nactual chunk sizes: %s\n\nactual chunk names: %s" % (
            chunk_size,
            str(candidate_chunk_lengths),
            str(actual_chunk_lengths_dict.keys()),
        )


def run_engine_test(
    arrays: List[np.ndarray], storage: StorageProvider, batched: bool, chunk_size: int
):
    key = TENSOR_KEY
    sample_count = 0

    # TODO: write test that tries to add_samples_to_tensor without create_tensor (should fail)
    create_tensor(key, storage, tensor_meta_from_array(arrays[0], batched, chunk_size))

    for i, a_in in enumerate(arrays):
        add_samples_to_tensor(
            a_in,
            key,
            storage,
            batched=batched,
        )

        a_in = normalize_and_batchify_shape(a_in, batched=batched)

        num_samples = a_in.shape[0]
        array_slice = slice(sample_count, sample_count + num_samples)
        a_out = read_samples_from_tensor(
            key=key, storage=storage, array_slice=array_slice
        )

        assert tensor_exists(key, storage), "Tensor {} was not found.".format(key)
        meta = read_tensor_meta(key, storage)

        sample_count += num_samples

        assert_meta_is_valid(
            meta,
            {
                "chunk_size": chunk_size,
                "length": sample_count,
                "dtype": a_in.dtype.name,
                "min_shape": tuple(a_in.shape[1:]),
                "max_shape": tuple(a_in.shape[1:]),
            },
        )

        assert np.array_equal(a_in, a_out), "Array not equal @ batch_index=%i." % i

    index_map = read_index_map(key, storage)
    assert_chunk_sizes(key, index_map, chunk_size, storage)


def benchmark_write(
    key, arrays, chunk_size, storage, batched, clear_memory_after_write=True
):
    create_tensor(key, storage, tensor_meta_from_array(arrays[0], batched, chunk_size))

    for a_in in arrays:
        add_samples_to_tensor(
            a_in,
            key,
            storage,
            batched=batched,
        )


def benchmark_read(key: str, storage: StorageProvider):
    read_samples_from_tensor(key, storage)
