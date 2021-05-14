from numpy import can_cast
import pytest

from hub.core.tests.common import (  # type: ignore
    parametrize_all_storages,
    current_test_name,
    parametrize_all_caches,
)
import pytest
from hub.constants import MB

NUM_FILES = 20


# helper functions for tests
def check_storage_provider(storage, key):
    FILE_1 = f"{key}_1"
    FILE_2 = f"{key}_2"

    storage[FILE_1] = b"hello world"
    assert storage[FILE_1] == b"hello world"
    assert storage.get_bytes(FILE_1, 2, 5) == b"llo"

    storage.set_bytes(FILE_1, b"abcde", 6)
    assert storage[FILE_1] == b"hello abcde"

    storage.set_bytes(FILE_1, b"tuvwxyz", 6)
    assert storage[FILE_1] == b"hello tuvwxyz"

    storage.set_bytes(FILE_2, b"hello world", 3)
    assert storage[FILE_2] == b"\x00\x00\x00hello world"
    storage.set_bytes(FILE_2, b"new_text", overwrite=True)
    assert storage[FILE_2] == b"new_text"

    assert len(storage) >= 1

    for _ in storage:
        pass

    del storage[FILE_1]
    del storage[FILE_2]

    with pytest.raises(KeyError):
        storage[FILE_1]
    storage.flush()


def check_cache(cache, key):
    chunk = b"0123456789123456" * MB
    assert cache.dirty_keys == set()
    assert set(cache.lru_sizes.keys()) == set()
    assert len(cache.cache_storage) == 0
    assert len(cache.next_storage) == 0
    assert cache.cache_used == 0
    assert len(cache) == 0

    FILE_1 = f"{key}_1"
    FILE_2 = f"{key}_2"
    FILE_3 = f"{key}_3"

    cache[FILE_1] = chunk
    assert cache.dirty_keys == {FILE_1}
    assert set(cache.lru_sizes.keys()) == {FILE_1}
    assert len(cache.cache_storage) == 1
    assert len(cache.next_storage) == 0
    assert cache.cache_used == 16 * MB
    assert len(cache) == 1

    cache[FILE_2] = chunk
    assert cache.dirty_keys == {FILE_1, FILE_2}
    assert set(cache.lru_sizes.keys()) == {FILE_1, FILE_2}
    assert len(cache.cache_storage) == 2
    assert len(cache.next_storage) == 0
    assert cache.cache_used == 32 * MB
    assert len(cache) == 2

    cache[FILE_3] = chunk
    assert cache.dirty_keys == {FILE_3, FILE_2}
    assert set(cache.lru_sizes.keys()) == {FILE_2, FILE_3}
    assert len(cache.cache_storage) == 2
    assert len(cache.next_storage) == 1
    assert cache.cache_used == 32 * MB
    assert len(cache) == 3

    cache[FILE_1]
    assert cache.dirty_keys == {FILE_3}
    assert set(cache.lru_sizes.keys()) == {FILE_1, FILE_3}
    assert len(cache.cache_storage) == 2
    assert len(cache.next_storage) == 2
    assert cache.cache_used == 32 * MB
    assert len(cache) == 3

    cache[FILE_3]
    assert cache.dirty_keys == {FILE_3}
    assert set(cache.lru_sizes.keys()) == {FILE_1, FILE_3}
    assert len(cache.cache_storage) == 2
    assert len(cache.next_storage) == 2
    assert cache.cache_used == 32 * MB
    assert len(cache) == 3

    del cache[FILE_3]
    assert cache.dirty_keys == set()
    assert set(cache.lru_sizes.keys()) == {FILE_1}
    assert len(cache.cache_storage) == 1
    assert len(cache.next_storage) == 2
    assert cache.cache_used == 16 * MB
    assert len(cache) == 2

    del cache[FILE_1]
    assert cache.dirty_keys == set()
    assert set(cache.lru_sizes.keys()) == set()
    assert len(cache.cache_storage) == 0
    assert len(cache.next_storage) == 1
    assert cache.cache_used == 0
    assert len(cache) == 1

    del cache[FILE_2]
    assert cache.dirty_keys == set()
    assert set(cache.lru_sizes.keys()) == set()
    assert len(cache.cache_storage) == 0
    assert len(cache.next_storage) == 0
    assert cache.cache_used == 0
    assert len(cache) == 0

    with pytest.raises(KeyError):
        cache[FILE_1]

    cache[FILE_1] = chunk
    assert cache.dirty_keys == {FILE_1}
    assert set(cache.lru_sizes.keys()) == {FILE_1}
    assert len(cache.cache_storage) == 1
    assert len(cache.next_storage) == 0
    assert cache.cache_used == 16 * MB
    assert len(cache) == 1

    cache[FILE_2] = chunk
    assert cache.dirty_keys == {FILE_1, FILE_2}
    assert set(cache.lru_sizes.keys()) == {FILE_1, FILE_2}
    assert len(cache.cache_storage) == 2
    assert len(cache.next_storage) == 0
    assert cache.cache_used == 32 * MB
    assert len(cache) == 2

    cache.flush()
    assert cache.dirty_keys == set()
    assert set(cache.lru_sizes.keys()) == {FILE_1, FILE_2}
    assert len(cache.cache_storage) == 2
    assert len(cache.next_storage) == 2
    assert cache.cache_used == 32 * MB
    assert len(cache) == 2

    del cache[FILE_1]
    del cache[FILE_2]

    assert cache.dirty_keys == set()
    assert set(cache.lru_sizes.keys()) == set()
    assert len(cache.cache_storage) == 0
    assert len(cache.next_storage) == 0
    assert cache.cache_used == 0
    assert len(cache) == 0


def write_to_files(storage, key):
    chunk = b"0123456789123456" * MB
    for i in range(NUM_FILES):
        storage[f"{key}_{i}"] = chunk
    storage.flush()


def read_from_files(storage, key):
    for i in range(NUM_FILES):
        storage[f"{key}_{i}"]


def delete_files(storage, key):
    for i in range(NUM_FILES):
        del storage[f"{key}_{i}"]


@parametrize_all_storages
def test_storage_provider(storage):
    key = current_test_name(with_uuid=True)
    check_storage_provider(storage, key)


@parametrize_all_caches
def test_cache(storage):
    key = current_test_name(with_uuid=True)
    check_storage_provider(storage, key)
    check_cache(storage, key)


@parametrize_all_storages
def test_storage_write_speeds(benchmark, storage):
    key = current_test_name(with_uuid=True)
    benchmark(write_to_files, storage, key)
    delete_files(storage, key)


@parametrize_all_caches
def test_cache_write_speeds(benchmark, storage):
    key = current_test_name(with_uuid=True)
    benchmark(write_to_files, storage, key)
    delete_files(storage, key)


@parametrize_all_storages
def test_storage_read_speeds(benchmark, storage):
    key = current_test_name(with_uuid=True)
    write_to_files(storage, key)
    benchmark(read_from_files, storage, key)
    delete_files(storage, key)


@parametrize_all_caches
def test_cache_read_speeds(benchmark, storage):
    key = current_test_name(with_uuid=True)
    write_to_files(storage, key)
    benchmark(read_from_files, storage, key)
    delete_files(storage, key)


@parametrize_all_caches
def test_full_cache_read_speeds(benchmark, storage):
    key = current_test_name(with_uuid=True)
    write_to_files(storage, key)
    read_from_files(storage, key)
    benchmark(read_from_files, storage, key)
    delete_files(storage, key)
