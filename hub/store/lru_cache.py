from collections import OrderedDict
from collections.abc import MutableMapping


class DummyLock:
    def __init__(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


Lock = DummyLock


class LRUCache(MutableMapping):
    def __init__(
        self,
        cache_storage: MutableMapping,
        actual_storage: MutableMapping,
        max_size,
    ):
        """Creates LRU cache using cache_storage and actual_storage containers
        max_size -> maximum cache size that is allowed
        """
        self._dirty = set()
        self._mutex = Lock()
        self._max_size = max_size
        self._cache_storage = cache_storage
        self._actual_storage = actual_storage
        self._total_cached = 0
        self._cached_items = OrderedDict()
        # assert len(self._cache_storage) == 0, "Initially cache storage should be empty"

    @property
    def cache_storage(self):
        """Storage which is used for caching
        Returns MutableMapping
        """
        return self._cache_storage

    @property
    def actual_storage(self):
        """Storage which is used for actual storing (not caching)
        Returns MutableMapping
        """
        return self._actual_storage

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _flush_dirty(self):
        for item in self._dirty:
            self._actual_storage[item] = self._cache_storage[item]
        self._dirty.clear()

    def flush(self):
        self._flush_dirty()
        if hasattr(self._cache_storage, "flush"):
            self._cache_storage.flush()
        if hasattr(self._actual_storage, "flush"):
            self._actual_storage.flush()

    def close(self):
        self._flush_dirty()
        if hasattr(self._cache_storage, "close"):
            self._cache_storage.close()
        if hasattr(self._actual_storage, "close"):
            self._actual_storage.close()

    def commit(self):
        self.close()

    def __getitem__(self, key):
        """ Gets item and puts it in the cache if not there """
        with self._mutex:
            if key in self._cached_items:
                self._cached_items.move_to_end(key)
                return self._cache_storage[key]
            else:
                result = self._actual_storage[key]
                self._free_memory(len(result))
                self._append_cache(key, result)
                return result

    def __setitem__(self, key, value):
        """ Sets item and puts it in the cache if not there"""
        with self._mutex:
            if key in self._cached_items:
                self._total_cached -= self._cached_items.pop(key)
            self._free_memory(len(value))
            self._append_cache(key, value)
            if key not in self._dirty:
                self._dirty.add(key)

    def __delitem__(self, key):
        deleted_from_cache = False
        with self._mutex:
            if key in self._cached_items:
                self._total_cached -= self._cached_items.pop(key)
                del self._cache_storage[key]
                self._dirty.discard(key)
                deleted_from_cache = True
            try:
                del self._actual_storage[key]
            except KeyError:
                if not deleted_from_cache:
                    raise

    def __len__(self):
        return len(
            self.actual_storage
        )  # TODO: In future might need to fix this to return proper len

    def __iter__(self):
        cached_keys = set(self._dirty)
        for i in self.actual_storage:
            cached_keys.discard(i)
            yield i
        yield from sorted(cached_keys)

    def _free_memory(self, extra_size):
        while (
            self._total_cached > 0 and extra_size + self._total_cached > self._max_size
        ):
            item, itemsize = self._cached_items.popitem(last=False)
            if item in self._dirty:
                self._actual_storage[item] = self._cache_storage[item]
                self._dirty.discard(item)
            del self._cache_storage[item]
            self._total_cached -= itemsize

    def _append_cache(self, key, value):
        self._total_cached += len(value)
        self._cached_items[key] = len(value)
        self._cache_storage[key] = value
