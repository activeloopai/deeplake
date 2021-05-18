from typing import Tuple

import numpy as np
import pytest
from hub.constants import GB, MB
from hub.core.chunk_engine import read_array, write_array
from hub.core.chunk_engine.tests.common import (CHUNK_SIZE_PARAM, CHUNK_SIZES,
                                                DTYPE_PARAM, DTYPES,
                                                NUM_BATCHES_PARAM, SHAPE_PARAM,
                                                TENSOR_KEY, get_random_array,
                                                parametrize_chunk_sizes,
                                                parametrize_dtypes,
                                                run_engine_test)
from hub.core.tests.common import (parametrize_all_caches,
                                   parametrize_all_storages)
from hub.core.typing import StorageProvider

BENCHMARK_NUM_BATCHES = (1,)
BENCHMARK_DTYPES = (
    "int64",
    "float64",
)
BENCHMARK_CHUNK_SIZES = (16 * MB,)
BENCHMARK_BATCHED_SHAPES = (
    # with int64/float64 = ~1GB
    (840, 224, 224, 3),
)


# parametrize decorators
parametrize_benchmark_chunk_sizes = pytest.mark.parametrize(
    CHUNK_SIZE_PARAM, BENCHMARK_CHUNK_SIZES
)
parametrize_benchmark_dtypes = pytest.mark.parametrize(DTYPE_PARAM, BENCHMARK_DTYPES)
parametrize_benchmark_shapes = pytest.mark.parametrize(
    SHAPE_PARAM, BENCHMARK_BATCHED_SHAPES
)
parametrize_benchmark_num_batches = pytest.mark.parametrize(
    NUM_BATCHES_PARAM, BENCHMARK_NUM_BATCHES
)


# TODO: full benchmarks (non-cache write/read)


def single_benchmark_write(info, key, arrays, chunk_size, storage, batched):
    actual_key = "%s_%i" % (key, info["iteration"])

    for a_in in arrays:
        write_array(
            a_in,
            actual_key,
            chunk_size,
            storage,
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


@pytest.mark.benchmark(group="chunk_engine_write_with_caches")
@parametrize_benchmark_shapes
@parametrize_benchmark_num_batches
@parametrize_benchmark_chunk_sizes
@parametrize_benchmark_dtypes
@parametrize_all_caches
def test_write_with_caches(
    benchmark,
    shape: Tuple[int],
    chunk_size: int,
    num_batches: int,
    dtype: str,
    storage: StorageProvider,
):
    benchmark_write(benchmark, shape, dtype, chunk_size, num_batches, storage)


@pytest.mark.full_benchmark
@pytest.mark.benchmark(group="chunk_engine_write_without_caches")
@parametrize_benchmark_shapes
@parametrize_benchmark_num_batches
@parametrize_benchmark_chunk_sizes
@parametrize_benchmark_dtypes
@parametrize_all_storages
def test_write_without_caches(
    benchmark,
    shape: Tuple[int],
    chunk_size: int,
    num_batches: int,
    dtype: str,
    storage: StorageProvider,
):
    benchmark_write(benchmark, shape, dtype, chunk_size, num_batches, storage)


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
