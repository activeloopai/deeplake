import pytest

from deeplake.core import LRUCache, MemoryProvider


def test_simple_caching():
    real_ds = MemoryProvider()
    real_ds["a/value"] = bytes("abcdefg", "utf-8")
    real_ds["a/other"] = bytes("123456789", "utf-8")
    cache_ds = MemoryProvider()

    lru_cache = LRUCache(cache_storage=cache_ds, next_storage=real_ds, cache_size=100)
    assert len(cache_ds) == 0

    assert str(lru_cache["a/value"], "utf-8") == "abcdefg"
    assert len(cache_ds) == 1

    assert str(lru_cache["a/other"], "utf-8") == "123456789"
    assert len(cache_ds) == 2

    with pytest.raises(KeyError):
        assert lru_cache["a/missing"] is None


def test_cache_expiring():
    real_ds = MemoryProvider()
    real_ds["a/five1"] = bytes("12345", "utf-8")
    real_ds["a/five2"] = bytes("12345", "utf-8")
    real_ds["a/one"] = bytes("1", "utf-8")
    real_ds["a/nine"] = bytes("123456789", "utf-8")
    real_ds["a/ten"] = bytes("1234567890", "utf-8")

    cache_ds = MemoryProvider()
    lru_cache = LRUCache(cache_storage=cache_ds, next_storage=real_ds, cache_size=10)

    assert list(cache_ds.dict.keys()) == []

    assert str(lru_cache["a/five1"], "utf-8") == "12345"
    assert str(lru_cache["a/five2"], "utf-8") == "12345"
    assert str(lru_cache["a/five2"], "utf-8") == "12345"
    assert str(lru_cache["a/five2"], "utf-8") == "12345"
    assert list(cache_ds.dict.keys()) == ["a/five1", "a/five2"]

    assert str(lru_cache["a/one"], "utf-8") == "1"
    assert list(cache_ds.dict.keys()) == ["a/five2", "a/one"]

    assert str(lru_cache["a/five1"], "utf-8") == "12345"
    assert list(cache_ds.dict.keys()) == ["a/one", "a/five1"]


def test_cache_zero_size():
    real_ds = MemoryProvider()
    real_ds["a/five1"] = bytes("12345", "utf-8")
    real_ds["a/five2"] = bytes("12345", "utf-8")
    real_ds["a/one"] = bytes("1", "utf-8")
    real_ds["a/nine"] = bytes("123456789", "utf-8")
    real_ds["a/ten"] = bytes("1234567890", "utf-8")

    cache_ds = MemoryProvider()
    lru_cache = LRUCache(cache_storage=cache_ds, next_storage=real_ds, cache_size=0)

    assert list(cache_ds.dict.keys()) == []

    assert str(lru_cache["a/five1"], "utf-8") == "12345"
    assert list(cache_ds.dict.keys()) == []

    assert str(lru_cache["a/five1"], "utf-8") == "12345"
    assert list(cache_ds.dict.keys()) == []


def test_get_bytes():
    real_ds = MemoryProvider()
    real_ds["a/five"] = bytes("12345", "utf-8")
    real_ds["a/ten"] = bytes("1234567890", "utf-8")

    cache_ds = MemoryProvider()
    lru_cache = LRUCache(cache_storage=cache_ds, next_storage=real_ds, cache_size=0)

    assert str(lru_cache.get_bytes("a/five", 0, 1), "utf-8") == "1"
    assert list(cache_ds.dict.keys()) == []

    for start in range(0, 9):
        assert (
            str(lru_cache.get_bytes("a/ten", start, start + 1), "utf-8")
            == f"{start + 1}"
        )

    assert list(cache_ds.dict.keys()) == []
