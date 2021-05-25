from typing import Tuple

import numpy as np
import pytest

from hub.constants import GB
from hub.core.chunk_engine import read_array, write_array
from hub.core.chunk_engine.tests.common import (
    get_random_array,
    TENSOR_KEY,
)
from hub.core.tests.common import parametrize_all_caches, parametrize_all_storages
from hub.core.typing import StorageProvider
from hub.tests.common_benchmark import (
    parametrize_benchmark_shapes,
    parametrize_benchmark_chunk_sizes,
    parametrize_benchmark_dtypes,
    parametrize_benchmark_num_batches,
)


def single_benchmark_write(info, key, arrays, chunk_size, storage, batched):
    actual_key = "%s_%i" % (key, info["iteration"])

    for a_in in arrays:
        write_array(
            array=a_in,
            key=actual_key,
            storage=storage,
            chunk_size=chunk_size,
            batched=batched,
        )

    info["iteration"] += 1

    return actual_key


def benchmark_write(benchmark, shape, dtype, chunk_size, num_batches, storage):
    """
    Benchmark `write_array`.

    Samples have FIXED shapes (must have the same shapes).
    Samples are provided WITH a batch axis.
    """

    arrays = [get_random_array(shape, dtype) for _ in range(num_batches)]

    gbs = (np.prod(shape) * num_batches * np.dtype(dtype).itemsize) / GB
    benchmark.extra_info["input_array_gigabytes"] = gbs

    info = {"iteration": 0}

    benchmark(
        single_benchmark_write,
        info,
        TENSOR_KEY,
        arrays,
        chunk_size,
        storage,
        batched=True,
    )

    storage.clear()


def single_benchmark_read(key, storage):
    read_array(key, storage)


def benchmark_read(benchmark, shape, dtype, chunk_size, num_batches, storage):
    """
    Benchmark `read_array`.

    Samples have FIXED shapes (must have the same shapes).
    Samples are provided WITH a batch axis.
    """

    arrays = [get_random_array(shape, dtype) for _ in range(num_batches)]

    info = {"iteration": 0}

    actual_key = single_benchmark_write(
        info, TENSOR_KEY, arrays, chunk_size, storage, batched=True
    )
    benchmark(single_benchmark_read, actual_key, storage)
    storage.clear()


# @pytest.mark.benchmark(group="chunk_engine_write_with_caches")
# @parametrize_benchmark_shapes
# @parametrize_benchmark_num_batches
# @parametrize_benchmark_chunk_sizes
# @parametrize_benchmark_dtypes
# @parametrize_all_caches
# def test_write_with_caches(
#     benchmark,
#     shape: Tuple[int],
#     chunk_size: int,
#     num_batches: int,
#     dtype: str,
#     storage: StorageProvider,
# ):
#     benchmark_write(
#         benchmark=benchmark,
#         shape=shape,
#         dtype=dtype,
#         chunk_size=chunk_size,
#         num_batches=num_batches,
#         storage=storage,
#     )


# @pytest.mark.full_benchmark
# @pytest.mark.benchmark(group="chunk_engine_write_without_caches")
# @parametrize_benchmark_shapes
# @parametrize_benchmark_num_batches
# @parametrize_benchmark_chunk_sizes
# @parametrize_benchmark_dtypes
# @parametrize_all_storages
# def test_write_without_caches(
#     benchmark,
#     shape: Tuple[int],
#     chunk_size: int,
#     num_batches: int,
#     dtype: str,
#     storage: StorageProvider,
# ):
#     benchmark_write(
#         benchmark=benchmark,
#         shape=shape,
#         dtype=dtype,
#         chunk_size=chunk_size,
#         num_batches=num_batches,
#         storage=storage,
#     )


@pytest.mark.benchmark(group="chunk_engine_read_with_caches")
@parametrize_benchmark_shapes
@parametrize_benchmark_num_batches
@parametrize_benchmark_chunk_sizes
@parametrize_benchmark_dtypes
@parametrize_all_caches
def test_read_with_caches(
    benchmark,
    shape: Tuple[int],
    chunk_size: int,
    num_batches: int,
    dtype: str,
    storage: StorageProvider,
):
    benchmark_read(benchmark, shape, dtype, chunk_size, num_batches, storage)


@pytest.mark.full_benchmark
@pytest.mark.benchmark(group="chunk_engine_read_without_caches")
@parametrize_benchmark_shapes
@parametrize_benchmark_num_batches
@parametrize_benchmark_chunk_sizes
@parametrize_benchmark_dtypes
@parametrize_all_storages
def test_read_without_caches(
    benchmark,
    shape: Tuple[int],
    chunk_size: int,
    num_batches: int,
    dtype: str,
    storage: StorageProvider,
):
    benchmark_read(benchmark, shape, dtype, chunk_size, num_batches, storage)
