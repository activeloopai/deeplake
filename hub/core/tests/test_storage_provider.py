import pytest
from hub.constants import MB
from hub.core.tests.common import parametrize_all_caches, parametrize_all_storages
from hub.tests.common import current_test_name
from numpy import can_cast

NUM_FILES = 20
KEY = "file"


# helper functions for tests
def check_storage_provider(storage):
    FILE_1 = f"{KEY}_1"
    FILE_2 = f"{KEY}_2"

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


def check_cache_state(cache, expected_state):
    assert cache.dirty_keys == expected_state[0]
    assert set(cache.lru_sizes.keys()) == expected_state[1]
    assert len(cache.cache_storage) == expected_state[2]
    assert len(cache.next_storage) == expected_state[3]
    assert cache.cache_used == expected_state[4]
    assert len(cache) == expected_state[5]


def check_cache(cache):
    chunk = b"0123456789123456" * MB
    FILE_1, FILE_2, FILE_3 = f"{KEY}_1", f"{KEY}_2", f"{KEY}_3"
    check_cache_state(cache, expected_state=[set(), set(), 0, 0, 0, 0])

    cache[FILE_1] = chunk
    check_cache_state(cache, expected_state=[{FILE_1}, {FILE_1}, 1, 0, 16 * MB, 1])

    cache[FILE_2] = chunk
    check_cache_state(
        cache, expected_state=[{FILE_1, FILE_2}, {FILE_1, FILE_2}, 2, 0, 32 * MB, 2]
    )

    cache[FILE_3] = chunk
    check_cache_state(
        cache, expected_state=[{FILE_3, FILE_2}, {FILE_3, FILE_2}, 2, 1, 32 * MB, 3]
    )

    cache[FILE_1]
    check_cache_state(
        cache, expected_state=[{FILE_3}, {FILE_1, FILE_3}, 2, 2, 32 * MB, 3]
    )

    cache[FILE_3]
    check_cache_state(
        cache, expected_state=[{FILE_3}, {FILE_1, FILE_3}, 2, 2, 32 * MB, 3]
    )

    del cache[FILE_3]
    check_cache_state(cache, expected_state=[set(), {FILE_1}, 1, 2, 16 * MB, 2])

    del cache[FILE_1]
    check_cache_state(cache, expected_state=[set(), set(), 0, 1, 0, 1])

    del cache[FILE_2]
    check_cache_state(cache, expected_state=[set(), set(), 0, 0, 0, 0])

    with pytest.raises(KeyError):
        cache[FILE_1]

    cache[FILE_1] = chunk
    check_cache_state(cache, expected_state=[{FILE_1}, {FILE_1}, 1, 0, 16 * MB, 1])

    cache[FILE_2] = chunk
    check_cache_state(
        cache, expected_state=[{FILE_1, FILE_2}, {FILE_1, FILE_2}, 2, 0, 32 * MB, 2]
    )

    cache.flush()
    check_cache_state(cache, expected_state=[set(), {FILE_1, FILE_2}, 2, 2, 32 * MB, 2])

    del cache[FILE_1]
    del cache[FILE_2]
    check_cache_state(cache, expected_state=[set(), set(), 0, 0, 0, 0])


@parametrize_all_storages
def test_storage_provider(storage):
    check_storage_provider(storage)


@parametrize_all_caches
def test_cache(storage):
    check_storage_provider(storage)
    check_cache(storage)
