from typing import Tuple

import pytest

from hub.core.tests.common import parametrize_all_caches, parametrize_all_storages
from hub.core.typing import StorageProvider
from hub.tests.common_benchmark import (
    parametrize_full_benchmark_shapes,
    parametrize_benchmark_chunk_sizes,
    parametrize_benchmark_dtypes,
    parametrize_benchmark_num_batches,
)
from .test_benchmark_chunk_engine import (
    benchmark_write,
    benchmark_read,
)


@pytest.mark.full_benchmark
@pytest.mark.benchmark(group="chunk_engine_write_with_caches_FULL")
@parametrize_full_benchmark_shapes
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
    benchmark_write(
        benchmark=benchmark,
        shape=shape,
        dtype=dtype,
        chunk_size=chunk_size,
        num_batches=num_batches,
        storage=storage,
    )


@pytest.mark.full_benchmark
@pytest.mark.benchmark(group="chunk_engine_write_without_caches_FULL")
@parametrize_full_benchmark_shapes
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
    benchmark_write(
        benchmark=benchmark,
        shape=shape,
        dtype=dtype,
        chunk_size=chunk_size,
        num_batches=num_batches,
        storage=storage,
    )


@pytest.mark.full_benchmark
@pytest.mark.benchmark(group="chunk_engine_read_with_caches_FULL")
@parametrize_full_benchmark_shapes
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
@pytest.mark.benchmark(group="chunk_engine_read_without_caches_FULL")
@parametrize_full_benchmark_shapes
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
