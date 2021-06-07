import pytest

from hub.constants import MB
from hub.tests.common_benchmark import (
    parametrize_benchmark_chunk_sizes,
    BENCHMARK_CHUNK_SIZES,
)
from hub.core.tests.common import parametrize_all_caches, parametrize_all_storages
from hub.core.storage.tests.test_storage_provider import KEY  # type: ignore

SIMULATED_DATA_SIZES = [128 * MB]

# caclulate the number of chunks needed for each entry in `SIMULATED_DATA_SIZES`
NUM_CHUNKS = []
for chunk_size in BENCHMARK_CHUNK_SIZES:
    for data_size in SIMULATED_DATA_SIZES:
        NUM_CHUNKS.append(data_size // chunk_size)

mark_cache_group = pytest.mark.benchmark(group="storage_with_caches")
mark_no_cache_group = pytest.mark.benchmark(group="storage_without_caches")

parametrize_benchmark_num_chunks = pytest.mark.parametrize("num_chunks", NUM_CHUNKS)


def write_to_files(storage, chunk_size, num_chunks):
    chunk = b"1" * chunk_size
    for i in range(num_chunks):
        storage[f"{KEY}_{i}"] = chunk
    storage.flush()


def read_from_files(storage, num_chunks):
    for i in range(num_chunks):
        storage[f"{KEY}_{i}"]


@mark_no_cache_group
@parametrize_all_storages
@parametrize_benchmark_chunk_sizes
@parametrize_benchmark_num_chunks
def test_storage_write_speeds(benchmark, storage, chunk_size, num_chunks):
    benchmark(write_to_files, storage, chunk_size, num_chunks)


@mark_cache_group
@parametrize_all_caches
@parametrize_benchmark_chunk_sizes
@parametrize_benchmark_num_chunks
def test_cache_write_speeds(benchmark, storage, chunk_size, num_chunks):
    benchmark(write_to_files, storage, chunk_size, num_chunks)


@mark_no_cache_group
@parametrize_all_storages
@parametrize_benchmark_chunk_sizes
@parametrize_benchmark_num_chunks
def test_storage_read_speeds(benchmark, storage, chunk_size, num_chunks):
    write_to_files(storage, chunk_size, num_chunks)
    benchmark(read_from_files, storage, num_chunks)


@mark_cache_group
@parametrize_all_caches
@parametrize_benchmark_chunk_sizes
@parametrize_benchmark_num_chunks
def test_cache_read_speeds(benchmark, storage, chunk_size, num_chunks):
    write_to_files(storage, chunk_size, num_chunks)
    benchmark(read_from_files, storage, num_chunks)


@mark_cache_group
@parametrize_all_caches
@parametrize_benchmark_chunk_sizes
@parametrize_benchmark_num_chunks
def test_full_cache_read_speeds(benchmark, storage, chunk_size, num_chunks):
    write_to_files(storage, chunk_size, num_chunks)
    read_from_files(storage, num_chunks)
    benchmark(read_from_files, storage, num_chunks)
