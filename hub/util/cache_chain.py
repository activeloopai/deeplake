from hub.core.caching.lru_cache import LRUCache
from typing import List
from hub.core.storage.provider import StorageProvider

# TODO proper exceptions
def get_cache_chain(cache_list: List[StorageProvider], size_list: List[int]):
    if not cache_list:
        raise Exception
    assert len(size_list) + 1 == len(cache_list)
    cache_list.reverse(), size_list.reverse()
    store = cache_list[0]
    for size, cache in zip(size_list, cache_list[1:]):
        store = LRUCache(cache, store, size)
    return store
