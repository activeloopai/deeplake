from hub.core.index import Index
import pytest

from typing import Dict, List

import numpy as np
import pytest

from hub.core.meta.index_map import read_index_map
from hub.core.meta.tensor_meta import read_tensor_meta, default_tensor_meta
from hub.core.tensor import (
    add_samples_to_tensor,
    create_tensor,
    tensor_exists,
    read_samples_from_tensor,
)
from hub.core.typing import StorageProvider
from hub.tests.common import TENSOR_KEY
from hub.util.array import (
    normalize_and_batchify_array_shape,
    normalize_and_batchify_shape,
)
from hub.util.keys import get_chunk_key

STORAGE_FIXTURE_NAME = "storage"
DATASET_FIXTURE_NAME = "ds"

MEMORY = "memory"
LOCAL = "local"
S3 = "s3"

ALL_PROVIDERS = [MEMORY, LOCAL, S3]

_ALL_CACHES_TUPLES = [(MEMORY, LOCAL), (MEMORY, S3), (LOCAL, S3), (MEMORY, LOCAL, S3)]
ALL_CACHES = list(map(lambda i: ",".join(i), _ALL_CACHES_TUPLES))

parametrize_all_storages = pytest.mark.parametrize(
    STORAGE_FIXTURE_NAME,
    ALL_PROVIDERS,
    indirect=True,
)

parametrize_all_caches = pytest.mark.parametrize(
    STORAGE_FIXTURE_NAME,
    ALL_CACHES,
    indirect=True,
)

parametrize_all_storages_and_caches = pytest.mark.parametrize(
    STORAGE_FIXTURE_NAME,
    ALL_PROVIDERS + ALL_CACHES,
    indirect=True,
)

parametrize_all_dataset_storages = pytest.mark.parametrize(
    DATASET_FIXTURE_NAME, ALL_PROVIDERS, indirect=True
)

parametrize_all_dataset_storages_and_caches = pytest.mark.parametrize(
    STORAGE_FIXTURE_NAME,
    ALL_PROVIDERS + ALL_CACHES,  # type: ignore
    indirect=True,
)


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
        ), "All chunks (except the last one) MUST be == `chunk_size`. chunk_size=%i\n\nactual chunk sizes: " "%s\n\nactual chunk names: %s" % (
            chunk_size,
            str(candidate_chunk_lengths),
            str(actual_chunk_lengths_dict.keys()),
        )


def run_engine_test(
    arrays: List[np.ndarray], storage: StorageProvider, batched: bool, chunk_size: int
):
    key = TENSOR_KEY
    sample_count = 0

    create_tensor(key, storage, default_tensor_meta(chunk_size=chunk_size))

    first_sample_shape = normalize_and_batchify_shape(arrays[0].shape, batched=batched)[
        1:
    ]
    expected_min_shape = first_sample_shape
    expected_max_shape = first_sample_shape

    for i, a_in in enumerate(arrays):
        add_samples_to_tensor(
            a_in,
            key,
            storage,
            batched=batched,
        )

        a_in = normalize_and_batchify_array_shape(a_in, batched=batched)

        current_batch_num_samples = a_in.shape[0]
        index = Index(slice(sample_count, sample_count + current_batch_num_samples))
        a_out = read_samples_from_tensor(key=key, storage=storage, index=index)

        assert tensor_exists(key, storage), "Tensor {} was not found.".format(key)
        meta = read_tensor_meta(key, storage)

        sample_count += current_batch_num_samples

        expected_min_shape = np.minimum(expected_min_shape, a_in.shape[1:])
        expected_max_shape = np.maximum(expected_max_shape, a_in.shape[1:])

        assert_meta_is_valid(
            meta,
            {
                "chunk_size": chunk_size,
                "length": sample_count,
                "dtype": a_in.dtype.name,
                "min_shape": list(expected_min_shape),
                "max_shape": list(expected_max_shape),
            },
        )

        assert np.array_equal(a_in, a_out), "Array not equal @ batch_index=%i." % i  # type: ignore

    index_map = read_index_map(key, storage)
    assert_chunk_sizes(key, index_map, chunk_size, storage)


def benchmark_write(
    key, arrays, chunk_size, storage, batched, clear_memory_after_write=True
):
    create_tensor(key, storage, default_tensor_meta(chunk_size=chunk_size))

    for a_in in arrays:
        add_samples_to_tensor(
            a_in,
            key,
            storage,
            batched=batched,
        )


def benchmark_read(key: str, storage: StorageProvider):
    read_samples_from_tensor(key, storage)
