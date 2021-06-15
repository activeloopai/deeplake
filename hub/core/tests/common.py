from hub.core.meta.tensor_meta import TensorMeta
from hub.core.index import Index
import pytest

from typing import Dict, List

import numpy as np
import pytest

from hub.core.meta.index_meta import IndexMeta
from hub.core.tensor import (
    append_tensor,
    create_tensor,
    extend_tensor,
    tensor_exists,
    read_samples_from_tensor,
)
from hub.core.typing import StorageProvider
from hub.tests.common import TENSOR_KEY
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


def assert_meta_is_valid(tensor_meta: TensorMeta, expected_meta: dict):
    for k, v in expected_meta.items():
        assert hasattr(tensor_meta, k), 'Key "%s" not found in meta: %s' % (
            k,
            str(tensor_meta),
        )
        assert v == getattr(
            tensor_meta, k
        ), 'Value for key "%s" mismatch.\n(actual): %s\n!=\n(expected):%s' % (
            k,
            getattr(tensor_meta, k),
            v,
        )


def assert_chunk_sizes(key: str, storage: StorageProvider, chunk_size: int):
    index_meta = IndexMeta.load(key, storage)

    incomplete_chunk_names = set()
    complete_chunk_count = 0
    total_chunks = 0
    actual_chunk_lengths_dict: Dict[str, int] = {}
    for i, entry in enumerate(index_meta.entries):
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

    create_tensor(key, storage, chunk_size=chunk_size, dtype=arrays[0].dtype.name)
    tensor_meta = TensorMeta.load(key, storage)

    first_sample_shape = arrays[0].shape
    expected_min_shape = first_sample_shape
    expected_max_shape = first_sample_shape

    for i, a_in in enumerate(arrays):
        if batched:
            current_batch_num_samples = a_in.shape[0]
            extend_tensor(a_in, key, storage, tensor_meta=tensor_meta)
        else:
            current_batch_num_samples = 1
            append_tensor(a_in, key, storage, tensor_meta=tensor_meta)

        index = Index(slice(sample_count, sample_count + current_batch_num_samples))
        a_out = read_samples_from_tensor(key=key, storage=storage, index=index)

        assert tensor_exists(key, storage), "Tensor {} was not found.".format(key)

        sample_count += current_batch_num_samples

        expected_min_shape = np.minimum(expected_min_shape, a_in.shape)
        expected_max_shape = np.maximum(expected_max_shape, a_in.shape)

        assert_meta_is_valid(
            tensor_meta,
            {
                "chunk_size": chunk_size,
                "length": sample_count,
                "dtype": a_in.dtype.name,
                "min_shape": list(expected_min_shape),
                "max_shape": list(expected_max_shape),
            },
        )

        assert np.array_equal(a_in, a_out), "Array not equal @ batch_index=%i." % i  # type: ignore

    assert_chunk_sizes(key, storage, chunk_size)


def benchmark_write(key, arrays, chunk_size, storage, batched):
    create_tensor(key, storage, chunk_size=chunk_size)

    for a_in in arrays:
        if batched:
            extend_tensor(a_in, key, storage)
        else:
            append_tensor(a_in, key, storage)


def benchmark_read(key: str, storage: StorageProvider):
    read_samples_from_tensor(key, storage)
