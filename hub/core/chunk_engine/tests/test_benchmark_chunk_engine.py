from typing import Tuple

import numpy as np
import pytest

from hub.constants import GB
from hub.core.tensor import (
    append_tensor,
    extend_tensor,
    read_samples_from_tensor,
    create_tensor,
)
from hub.core.typing import StorageProvider
from hub.tests.common import TENSOR_KEY, get_random_array
from hub.tests.common_benchmark import (
    parametrize_benchmark_chunk_sizes,
    parametrize_benchmark_dtypes,
    parametrize_benchmark_num_batches,
    parametrize_benchmark_shapes,
)


def single_benchmark_write(info, key, arrays, chunk_size, storage, batched):
    actual_key = "%s_%i" % (key, info["iteration"])

    create_tensor(
        actual_key, storage, chunk_size=chunk_size, dtype=arrays[0].dtype.name
    )

    for a_in in arrays:
        if batched:
            extend_tensor(a_in, actual_key, storage)
        else:
            append_tensor(a_in, actual_key, storage)

    info["iteration"] += 1

    return actual_key


def benchmark_write(benchmark, shape, dtype, chunk_size, num_batches, storage):
    """
    Benchmark `add_samples_to_tensor`.

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
    read_samples_from_tensor(key, storage)


def benchmark_read(benchmark, shape, dtype, chunk_size, num_batches, storage):
    """
    Benchmark `read_samples_from_tensor`.

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


@pytest.mark.benchmark(group="chunk_engine_write_memory")
@parametrize_benchmark_shapes
@parametrize_benchmark_num_batches
@parametrize_benchmark_chunk_sizes
@parametrize_benchmark_dtypes
def test_write_memory(
    benchmark,
    shape: Tuple[int],
    chunk_size: int,
    num_batches: int,
    dtype: str,
    memory_storage: StorageProvider,
):
    benchmark_write(
        benchmark=benchmark,
        shape=shape,
        dtype=dtype,
        chunk_size=chunk_size,
        num_batches=num_batches,
        storage=memory_storage,
    )


@pytest.mark.benchmark(group="chunk_engine_read_memory")
@parametrize_benchmark_shapes
@parametrize_benchmark_num_batches
@parametrize_benchmark_chunk_sizes
@parametrize_benchmark_dtypes
def test_read_memory(
    benchmark,
    shape: Tuple[int],
    chunk_size: int,
    num_batches: int,
    dtype: str,
    memory_storage: StorageProvider,
):
    benchmark_read(benchmark, shape, dtype, chunk_size, num_batches, memory_storage)
