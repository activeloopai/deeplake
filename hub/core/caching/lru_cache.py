from collections import OrderedDict
from multiprocessing import Lock
from hub.core.storage.provider import Provider


class DummyLock:
    def __init__(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class LRUCache(Provider):
    def __init__(
        self,
        cache_storage: Provider,
        next_storage: Provider,
        cache_size,
    ):
        self._next_storage = next_storage
        self._cache_storage = cache_storage
        self._cache_size = cache_size  # max size of cache_storage
        self._dirty_keys = set()  # keys stored in cache but not actual storage
        self._lock = DummyLock()  # TODO actual lock after testing
        self._cache_used = 0  # size of cache used
        self._lru_lengths = OrderedDict()  # tracks key order and length of value

    def flush(self):
        for item in self._dirty_keys:
            self._next_storage[item] = self._cache_storage[item]
        self._dirty_keys.clear()

        # if next_storage is also a cache
        if hasattr(self._next_storage, "flush"):
            self._next_storage.flush()

    def __getitem__(self, key):
        """Gets item and puts it in the cache if not there"""
        with self._lock:
            # already exists, move to end i.e. refresh position for LRU
            if key in self._lru_lengths:
                self._lru_lengths.move_to_end(key)
                return self._cache_storage[key]
            else:
                result = self._next_storage[key]
                # only insert in cache if it can fit
                if len(result) < self._cache_size:
                    self._free_up_space(len(result))
                    self._insert_in_cache(key, result)
                return result

    def __setitem__(self, key, value):
        """Sets item and puts it in the cache if not there"""
        with self._lock:
            if key in self._lru_lengths:
                self._cache_used -= self._lru_lengths.pop(key)
            if len(value) < self._cache_size:
                self._free_up_space(len(value))
                self._insert_in_cache(key, value)
                if key not in self._dirty_keys:
                    self._dirty_keys.add(key)
            # value is larger than cache, directly set it in next layer
            else:
                if key in self._dirty_keys:
                    self._dirty_keys.discard(key)
                self._next_storage[key] = value

    def __delitem__(self, key):
        with self._lock:
            deleted_from_cache = False
            if key in self._lru_lengths:
                self._cache_used -= self._lru_lengths.pop(key)
                del self._cache_storage[key]
                self._dirty_keys.discard(key)
                deleted_from_cache = True
            try:
                del self._next_storage[key]
            except KeyError:
                if not deleted_from_cache:
                    raise

    def _free_up_space(self, extra_size):
        # keep on freeing space in cache till extra_size can fit in
        while self._cache_used > 0 and extra_size + self._cache_used > self._cache_size:
            item, itemsize = self._lru_lengths.popitem(last=False)
            if item in self._dirty_keys:
                self._next_storage[item] = self._cache_storage[item]
                self._dirty_keys.discard(item)
            del self._cache_storage[item]
            self._cache_used -= itemsize

    def _insert_in_cache(self, key, value):
        self._cache_storage[key] = value
        self._cache_used += len(value)
        self._lru_lengths[key] = len(value)

    def __len__(self):
        return len(self.actual_storage)

    def __iter__(self):
        cached_keys = self._dirty_keys.copy()
        for i in self.actual_storage:
            cached_keys.discard(i)
            yield i
        yield from sorted(cached_keys)
