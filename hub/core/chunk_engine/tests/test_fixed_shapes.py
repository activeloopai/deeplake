from typing import Tuple

import numpy as np
import pytest
from hub.constants import GB, MB
from hub.core.chunk_engine.tests.common import (CHUNK_SIZES, DTYPES,
                                                TENSOR_KEY, benchmark_read,
                                                benchmark_write,
                                                get_random_array,
                                                run_engine_test)
from hub.core.tests.common import (parametrize_all_caches,
                                   parametrize_all_storages_and_caches)
from hub.core.typing import StorageProvider
from hub.tests.common import current_test_name

np.random.seed(1)


# number of batches (unbatched implicitly = 1 sample per batch) per test
NUM_BATCHES = (1,)


UNBATCHED_SHAPES = (
    (1,),
    (100,),
    (1, 1, 3),
    (20, 90),
    (3, 28, 24, 1),
)


BATCHED_SHAPES = (
    (1, 1),
    (10, 1),
    (1, 30, 30),
    (3, 3, 12, 12, 1),
)

SHAPE_PARAM = "shape"
NUM_BATCHES_PARAM = "num_batches"
DTYPE_PARAM = "dtype"
CHUNK_SIZE_PARAM = "chunk_size"

parametrize_chunk_sizes = pytest.mark.parametrize(CHUNK_SIZE_PARAM, CHUNK_SIZES)
parametrize_dtypes = pytest.mark.parametrize(DTYPE_PARAM, DTYPES)


# for benchmarks
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


parametrize_benchmark_chunk_sizes = pytest.mark.parametrize(
    CHUNK_SIZE_PARAM, BENCHMARK_CHUNK_SIZES
)
parametrize_benchmark_dtypes = pytest.mark.parametrize(DTYPE_PARAM, BENCHMARK_DTYPES)


@pytest.mark.parametrize(SHAPE_PARAM, UNBATCHED_SHAPES)
@pytest.mark.parametrize(NUM_BATCHES_PARAM, NUM_BATCHES)
@parametrize_chunk_sizes
@parametrize_dtypes
@parametrize_all_storages_and_caches
def test_unbatched(
    shape: Tuple[int],
    chunk_size: int,
    num_batches: int,
    dtype: str,
    storage: StorageProvider,
):
    """
    Samples have FIXED shapes (must have the same shapes).
    Samples are provided WITHOUT a batch axis.
    """

    arrays = [get_random_array(shape, dtype) for _ in range(num_batches)]
    run_engine_test(arrays, storage, batched=False, chunk_size=chunk_size)


@pytest.mark.parametrize(SHAPE_PARAM, BATCHED_SHAPES)
@pytest.mark.parametrize(NUM_BATCHES_PARAM, NUM_BATCHES)
@parametrize_chunk_sizes
@parametrize_dtypes
@parametrize_all_storages_and_caches
def test_batched(
    shape: Tuple[int],
    chunk_size: int,
    num_batches: int,
    dtype: str,
    storage: StorageProvider,
):
    """
    Samples have FIXED shapes (must have the same shapes).
    Samples are provided WITH a batch axis.
    """

    arrays = [get_random_array(shape, dtype) for _ in range(num_batches)]
    run_engine_test(arrays, storage, batched=True, chunk_size=chunk_size)


@pytest.mark.benchmark(group="chunk_engine")
@pytest.mark.parametrize(SHAPE_PARAM, BENCHMARK_BATCHED_SHAPES)
@pytest.mark.parametrize(NUM_BATCHES_PARAM, BENCHMARK_NUM_BATCHES)
@parametrize_benchmark_chunk_sizes
@parametrize_benchmark_dtypes
@parametrize_all_caches
def test_write(
    benchmark,
    shape: Tuple[int],
    chunk_size: int,
    num_batches: int,
    dtype: str,
    storage: StorageProvider,
):
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
        benchmark_write,
        info,
        TENSOR_KEY,
        arrays,
        chunk_size,
        storage,
        batched=True,
    )


@pytest.mark.benchmark(group="chunk_engine")
@pytest.mark.parametrize(SHAPE_PARAM, BENCHMARK_BATCHED_SHAPES)
@pytest.mark.parametrize(NUM_BATCHES_PARAM, BENCHMARK_NUM_BATCHES)
@parametrize_benchmark_chunk_sizes
@parametrize_benchmark_dtypes
@parametrize_all_caches
def test_read(
    benchmark,
    shape: Tuple[int],
    chunk_size: int,
    num_batches: int,
    dtype: str,
    storage: StorageProvider,
):
    """
    Benchmark `read_array`.

    Samples have FIXED shapes (must have the same shapes).
    Samples are provided WITH a batch axis.
    """

    arrays = [get_random_array(shape, dtype) for _ in range(num_batches)]

    info = {"iteration": 0}

    actual_key = benchmark_write(
        info, TENSOR_KEY, arrays, chunk_size, storage, batched=True
    )
    benchmark(benchmark_read, actual_key, storage)
