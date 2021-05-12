import pytest

import numpy as np

from hub.core.chunk_engine.tests.common import (
    run_engine_test,
    CHUNK_SIZES,
    DTYPES,
    get_random_array,
    STORAGE_PROVIDERS,
)


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


# for benchmarks
BENCHMARK_NUM_BATCHES = (1,)
BENCHMARK_DTYPES = ("int64", "float64")
BENCHMARK_CHUNK_SIZES = (
    16000000,  # 16MB
    3000000000,  # 3GB
)
BENCHMARK_BATCHED_SHAPES = (
    # with int64/float64 = ~1GB
    (100, 224, 224, 3),
    # with int64/float64 = ~2GB
    # (200, 224, 224, 3),
)


@pytest.mark.parametrize("shape", UNBATCHED_SHAPES)
@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
@pytest.mark.parametrize("num_batches", NUM_BATCHES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("storage", STORAGE_PROVIDERS)
def test_unbatched(shape, chunk_size, num_batches, dtype, storage):
    """
    Samples have FIXED shapes (must have the same shapes).
    Samples are provided WITHOUT a batch axis.
    """

    arrays = [get_random_array(shape, dtype) for _ in range(num_batches)]
    run_engine_test(arrays, storage, batched=False, chunk_size=chunk_size)


@pytest.mark.parametrize("shape", BATCHED_SHAPES)
@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
@pytest.mark.parametrize("num_batches", NUM_BATCHES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("storage", STORAGE_PROVIDERS)
def test_batched(shape, chunk_size, num_batches, dtype, storage):
    """
    Samples have FIXED shapes (must have the same shapes).
    Samples are provided WITH a batch axis.
    """

    arrays = [get_random_array(shape, dtype) for _ in range(num_batches)]
    run_engine_test(arrays, storage, batched=True, chunk_size=chunk_size)


@pytest.mark.parametrize("shape", BENCHMARK_BATCHED_SHAPES)
@pytest.mark.parametrize("chunk_size", BENCHMARK_CHUNK_SIZES)
@pytest.mark.parametrize("num_batches", BENCHMARK_NUM_BATCHES)
@pytest.mark.parametrize("dtype", BENCHMARK_DTYPES)
@pytest.mark.parametrize("storage", STORAGE_PROVIDERS)
def test_benchmark_batched(benchmark, shape, chunk_size, num_batches, dtype, storage):
    """
    Benchmark with larger arrays.

    Samples have FIXED shapes (must have the same shapes).
    Samples are provided WITH a batch axis.
    """

    arrays = [get_random_array(shape, dtype) for _ in range(num_batches)]
    benchmark(run_engine_test, arrays, storage, batched=True, chunk_size=chunk_size)
