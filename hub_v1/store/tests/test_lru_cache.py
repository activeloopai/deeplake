"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

from hub_v1.store.lru_cache import LRUCache

import zarr


def test_lru_cache():
    data = bytes("Hello World", "utf-8")
    cache = LRUCache(zarr.MemoryStore(), zarr.MemoryStore(), 30)
    cache["Aello"] = data
    cache["Beta"] = data
    assert "Aello" in cache._cached_items
    assert "Beta" in cache._cached_items
    assert "Aello" in cache.cache_storage
    assert "Beta" in cache.cache_storage
    cache["Gamma"] = data
    cache["Gamma"] = data
    assert "Aello" not in cache._cached_items
    assert "Aello" not in cache.cache_storage
    assert "Gamma" in cache._cached_items
    assert "Gamma" in cache.cache_storage

    assert list(sorted(cache)) == ["Aello", "Beta", "Gamma"]
    assert list(sorted(cache.cache_storage)) == ["Beta", "Gamma"]
    assert list(sorted(cache.actual_storage)) == ["Aello"]
    del cache["Gamma"]
    assert list(sorted(cache)) == ["Aello", "Beta"]
    assert list(sorted(cache.cache_storage)) == ["Beta"]
    cache["Aello"]
    cache["Beta"]
    try:
        del cache["KeyError"]
    except KeyError:
        pass
    assert list(sorted(cache.actual_storage)) == ["Aello"]
    cache.flush()
    assert list(sorted(cache.actual_storage)) == ["Aello", "Beta"]
    cache.commit()


if __name__ == "__main__":
    test_lru_cache()
