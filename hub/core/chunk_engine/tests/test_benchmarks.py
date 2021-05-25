from typing import Tuple

import numpy as np
import pytest

from hub.constants import GB, MB
from hub.core.chunk_engine import read_array, write_array
from hub.core.chunk_engine.tests.common import get_random_array
from hub.core.tests.common import (
    parametrize_all_caches,
)
from hub.core.typing import StorageProvider
from hub.tests.common import (
    CHUNK_SIZE_PARAM,
    DTYPE_PARAM,
    NUM_BATCHES_PARAM,
    SHAPE_PARAM,
    TENSOR_KEY,
)

BENCHMARK_NUM_BATCHES = (1,)
BENCHMARK_DTYPES = (
    "int64",
    "float64",
)
BENCHMARK_CHUNK_SIZES = (16 * MB,)
BENCHMARK_BATCHED_SHAPES = (
    # with int64/float64 = ~1GB
    # (840, 224, 224, 3),
    (5, 224, 224, 3),
)


parametrize_benchmark_chunk_sizes = pytest.mark.parametrize(
    CHUNK_SIZE_PARAM, BENCHMARK_CHUNK_SIZES
)
parametrize_benchmark_dtypes = pytest.mark.parametrize(DTYPE_PARAM, BENCHMARK_DTYPES)


# TODO: full benchmarks (non-cache write/read)


def benchmark_write(info, key, arrays, chunk_size, storage, batched):
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


def benchmark_read(key: str, storage: StorageProvider):
    read_array(key, storage)


@pytest.mark.benchmark(group="chunk_engine")
@pytest.mark.parametrize(SHAPE_PARAM, BENCHMARK_BATCHED_SHAPES)
@pytest.mark.parametrize(NUM_BATCHES_PARAM, BENCHMARK_NUM_BATCHES)
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
def test_read_with_caches(
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
