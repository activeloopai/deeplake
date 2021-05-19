import pytest

from hub.constants import MB
from hub.core.tests.common import parametrize_all_caches, parametrize_all_storages
from hub.core.storage.tests.test_storage_provider import KEY


NUM_CHUNKS = 63


mark_cache_group = pytest.mark.benchmark(group="storage_with_caches")
mark_no_cache_group = pytest.mark.benchmark(group="storage_without_caches")


def write_to_files(storage):
    chunk = b"0123456789123456" * MB
    for i in range(NUM_CHUNKS):
        storage[f"{KEY}_{i}"] = chunk
    storage.flush()


def read_from_files(storage):
    for i in range(NUM_CHUNKS):
        storage[f"{KEY}_{i}"]


@mark_no_cache_group
@parametrize_all_storages
def test_storage_write_speeds(benchmark, storage):
    benchmark(write_to_files, storage)


@mark_cache_group
@parametrize_all_caches
def test_cache_write_speeds(benchmark, storage):
    benchmark(write_to_files, storage)


@mark_no_cache_group
@parametrize_all_storages
def test_storage_read_speeds(benchmark, storage):
    write_to_files(storage)
    benchmark(read_from_files, storage)


@mark_cache_group
@parametrize_all_caches
def test_cache_read_speeds(benchmark, storage):
    write_to_files(storage)
    benchmark(read_from_files, storage)


@mark_cache_group
@parametrize_all_caches
def test_full_cache_read_speeds(benchmark, storage):
    write_to_files(storage)
    read_from_files(storage)
    benchmark(read_from_files, storage)
