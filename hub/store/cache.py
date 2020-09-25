import os
import zarr

from collections import OrderedDict


class CacheStore(zarr.LMDBStore):
    def __init__(self, path, buffers=True, **kwargs):
        super(CacheStore, self).__init__(path, buffers=True, **kwargs)
        # orders are within single process
        # TODO make _order variable be inside db
        self._order = OrderedDict()

    def move_to_end(self, key):
        """Move key to the end"""
        # FIXME Zarr sometimes inserts tuple, but lmbd can't have tuple key (".zgroup", "z.group")
        if isinstance(key, tuple):
            key = key[0]
        self._order.move_to_end(key)

    def popitem(self, last=False):
        """Remove the first value from the cache, as this will be the least recently"""
        k, v = self._order.popitem(last=last)
        return k, self.pop(k)

    def pop(self, key):
        """Remove an element from the cache"""
        if isinstance(key, tuple):
            key = key[0]  # Zarr sometimes inserts tuple?
        if key in self._order:
            self._order.pop(key)
        el = self[key]
        del self[key]
        return el

    def __setitem__(self, key, value):
        """On each new add, remember the order"""
        self._order[key] = key
        if isinstance(key, tuple):
            key = key[0]
        super().__setitem__(key, value)

    def __getitem__(self, key):
        """On each new add, remember the order"""
        if isinstance(key, tuple):
            key = key[0]
        el = super().__getitem__(key)
        return el

    def __delitem__(self, key):
        if isinstance(key, tuple):
            key = key[0]
        super().__delitem__(key)


class Cache(zarr.LRUStoreCache):
    def __init__(self, store, max_size, path="~/.activeloop/cache"):
        """
        Extends zarr.LRUStoreCache with LMBD Cache that could be shared across

        Storage class that implements a least-recently-used (LRU) cache layer over
        some other store. Intended primarily for use with stores that can be slow to
        access, e.g., remote stores that require network communication to store and
        retrieve data.

        Parameters
        ----------
        store : MutableMapping
            The store containing the actual data to be cached.
        max_size : int
            The maximum size that the cache may grow to, in number of bytes. Provide `None`
            if you would like the cache to have unlimited size.
        """
        super(Cache, self).__init__(store, max_size)
        self.path = os.path.expanduser(path)
        os.makedirs(self.path, exist_ok=True)
        self._values_cache = CacheStore(self.path, buffers=True)
        self.cache_key = "_current_size"

    @property
    def _current_size(self):
        """ get size counter from the cache """
        if "_values_cache" not in dir(self) or self.cache_key not in self._values_cache:
            return 0
        return int.from_bytes(
            self._values_cache[self.cache_key], byteorder="big", signed=True
        )

    @_current_size.setter
    def _current_size(self, x):
        """ set size counter to the cache """
        if "_values_cache" not in dir(self):
            return
        self._values_cache[self.cache_key] = int.to_bytes(
            x, length=32, byteorder="big", signed=True
        )

    def commit(self):
        """ closes the cache db """
        self._values_cache.close()