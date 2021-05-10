from hub.core.caching.lru_cache import LRUCache
from typing import List
from hub.core.storage.provider import Provider


def get_cache_chain(cache_list: List[Provider], size_list: List[int]):
    if not cache_list:
        raise Exception
    cache_list.reverse()
    size_list.reverse()
    cache = cache_list[0]
    for i, item in enumerate(cache_list[1:]):
        cache = LRUCache(item, cache, size_list[i])
    return cache
