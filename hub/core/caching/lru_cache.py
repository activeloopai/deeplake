from collections import OrderedDict
from hub.core.storage.provider import StorageProvider
from typing import Set

# TODO use lock for multiprocessing
class LRUCache(StorageProvider):
    def __init__(
        self,
        cache_storage: StorageProvider,
        next_storage: StorageProvider,
        cache_size: int,
    ):
        self._next_storage = next_storage
        self._cache_storage = cache_storage
        self._cache_size = cache_size  # max size of cache_storage

        # LRU state variables
        self._dirty_keys: Set[str] = set()  # keys in cache but not next_storage
        self._cache_used = 0  # size of cache used
        # tracks keys in lru order, stores size of value, only keys present in this exist in cache
        self._lru_sizes: OrderedDict[str, int] = OrderedDict()

    def flush(self):
        # writing all keys in cache but not in next_storage
        for item in self._dirty_keys:
            self._next_storage[item] = self._cache_storage[item]
        self._dirty_keys.clear()
        self._next_storage.flush()

    def __getitem__(self, key):
        """Gets item and puts it in the cache if not there"""
        if key in self._lru_sizes:
            self._lru_sizes.move_to_end(key)  # refresh position for LRU
            return self._cache_storage[key]
        else:
            result = self._next_storage[key]  # fetch from storage
            if len(result) <= self._cache_size:  # insert in cache if it fits
                self._insert_in_cache(key, result)
            return result

    def __setitem__(self, key, value):
        """Sets item and puts it in the cache if not there"""
        if key in self._lru_sizes:
            size = self._lru_sizes.pop(key)
            self._cache_used -= size

        if len(value) <= self._cache_size:
            self._insert_in_cache(key, value)
            self._dirty_keys.add(key)
        else:  # larger than cache, directly send to next layer
            self._dirty_keys.discard(key)
            self._next_storage[key] = value

    def __delitem__(self, key):
        deleted_from_cache = False
        if key in self._lru_sizes:
            size = self._lru_sizes.pop(key)
            self._cache_used -= size
            del self._cache_storage[key]
            self._dirty_keys.discard(key)
            deleted_from_cache = True

        try:
            del self._next_storage[key]
        except KeyError:
            if not deleted_from_cache:
                raise

    def __len__(self):
        return len(self._list_keys())

    def __iter__(self):
        yield from self._list_keys()

    def _free_up_space(self, extra_size):
        # keep on freeing space in cache till extra_size can fit in
        while self._cache_used > 0 and extra_size + self._cache_used > self._cache_size:
            self._pop_from_cache()

    def _pop_from_cache(self):
        # removes the item that was used the longest time back
        item, itemsize = self._lru_sizes.popitem(last=False)
        if item in self._dirty_keys:
            self._next_storage[item] = self._cache_storage[item]
            self._dirty_keys.discard(item)
        del self._cache_storage[item]
        self._cache_used -= itemsize

    def _insert_in_cache(self, key, value):
        # adds key value pair to cache
        self._free_up_space(len(value))
        self._cache_storage[key] = value
        self._cache_used += len(value)
        self._lru_sizes[key] = len(value)

    def _list_keys(self):
        all_keys = {item for item in self._next_storage}
        for item in self._cache_storage:
            all_keys.add(item)
        return all_keys
