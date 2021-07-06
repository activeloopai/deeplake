from collections import OrderedDict
from hub.core.storage.cachable import Cachable
from typing import Callable, Set, Union

from hub.core.storage.provider import StorageProvider


# TODO use lock for multiprocessing
class LRUCache(StorageProvider):
    """LRU Cache that uses StorageProvider for caching"""

    def __init__(
        self,
        cache_storage: StorageProvider,
        next_storage: StorageProvider,
        cache_size: int,
    ):
        """Initializes the LRUCache. It can be chained with other LRUCache objects to create multilayer caches.

        Args:
            cache_storage (StorageProvider): The storage being used as the caching layer of the cache.
                This should be a base provider such as MemoryProvider, LocalProvider or S3Provider but not another LRUCache.
            next_storage (StorageProvider): The next storage layer of the cache.
                This can either be a base provider (i.e. it is the final storage) or another LRUCache (i.e. in case of chained cache).
                While reading data, all misses from cache would be retrieved from here.
                While writing data, the data will be written to the next_storage when cache_storage is full or flush is called.
            cache_size (int): The total space that can be used from the cache_storage in bytes.
                This number may be less than the actual space available on the cache_storage.
                Setting it to a higher value than actually available space may lead to unexpected behaviors.
        """
        self.next_storage = next_storage
        self.cache_storage = cache_storage
        self.cache_size = cache_size

        # tracks keys in lru order, stores size of value, only keys present in this exist in cache
        self.lru_sizes: OrderedDict[str, int] = OrderedDict()
        self.dirty_keys: Set[str] = set()  # keys present in cache but not next_storage
        self.cache_used = 0

    def flush(self):
        """Writes data from cache_storage to next_storage. Only the dirty keys are written.
        This is a cascading function and leads to data being written to the final storage in case of a chained cache.
        """
        self.check_readonly()
        for key in self.dirty_keys.copy():
            self._forward(key)
        self.next_storage.flush()

    def get_cachable(self, path: str, expected_class):
        """If the data at `path` was stored using the output of a `Cachable` object's `tobytes` function,
        this function will read it back into object form & keep the object in cache.

        Args:
            path (str): Path to the stored cachable.
            expected_class (callable): The expected subclass of `Cachable`.

        Raises:
            ValueError: If the incorrect `expected_class` was provided.
            ValueError: If the type of the data at `path` is invalid.

        Returns:
            An instance of `expected_class` populated with the data.
        """

        item = self[path]

        if isinstance(item, Cachable):
            if type(item) != expected_class:
                raise ValueError(
                    f"'{path}' was expected to have the class '{expected_class.__name__}'. Instead, got: '{type(item)}'."
                )
            return item

        if isinstance(item, (bytes, memoryview)):
            obj = expected_class.frombuffer(item)
            if len(obj) <= self.cache_size:
                self._insert_in_cache(path, obj)
            return obj

        raise ValueError(f"Item at '{path}' got an invalid type: '{type(item)}'.")

    def __getitem__(self, path: str):
        """If item is in cache_storage, retrieves from there and returns.
        If item isn't in cache_storage, retrieves from next storage, stores in cache_storage (if possible) and returns.

        Args:
            path (str): The path relative to the root of the underlying storage.

        Raises:
            KeyError: if an object is not found at the path.

        Returns:
            bytes: The bytes of the object present at the path.
        """
        if path in self.lru_sizes:
            self.lru_sizes.move_to_end(path)  # refresh position for LRU
            return self.cache_storage[path]
        else:
            result = self.next_storage[path]  # fetch from storage, may throw KeyError
            if len(result) <= self.cache_size:  # insert in cache if it fits
                self._insert_in_cache(path, result)
            return result

    def __setitem__(self, path: str, value: Union[bytes, Cachable]):
        """Puts the item in the cache_storage (if possible), else writes to next_storage.

        Args:
            path (str): the path relative to the root of the underlying storage.
            value (bytes): the value to be assigned at the path.

        Raises:
            ReadOnlyError: If the provider is in read-only mode.
        """
        self.check_readonly()
        if path in self.lru_sizes:
            size = self.lru_sizes.pop(path)
            self.cache_used -= size

        if len(value) <= self.cache_size:
            self._insert_in_cache(path, value)
            self.dirty_keys.add(path)
        else:  # larger than cache, directly send to next layer
            self._forward_value(path, value)

        self.maybe_flush()

    def __delitem__(self, path: str):
        """Deletes the object present at the path from the cache and the underlying storage.

        Args:
            path (str): the path to the object relative to the root of the provider.

        Raises:
            KeyError: If an object is not found at the path.
            ReadOnlyError: If the provider is in read-only mode.
        """
        self.check_readonly()
        deleted_from_cache = False
        if path in self.lru_sizes:
            size = self.lru_sizes.pop(path)
            self.cache_used -= size
            del self.cache_storage[path]
            self.dirty_keys.discard(path)
            deleted_from_cache = True

        try:
            del self.next_storage[path]
        except KeyError:
            if not deleted_from_cache:
                raise

        self.maybe_flush()

    def clear_cache(self):
        """Flushes the content of the cache and and then deletes contents of all the layers of it.
        This doesn't delete data from the actual storage.
        """
        self.check_readonly()
        self.flush()
        self.cache_used = 0
        self.lru_sizes.clear()
        self.dirty_keys.clear()
        self.cache_storage.clear()

        if hasattr(self.next_storage, "clear_cache"):
            self.next_storage.clear_cache()

    def clear(self):
        """Deletes ALL the data from all the layers of the cache and the actual storage.
        This is an IRREVERSIBLE operation. Data once deleted can not be recovered.
        """
        self.check_readonly()
        self.cache_used = 0
        self.lru_sizes.clear()
        self.dirty_keys.clear()
        self.cache_storage.clear()
        self.next_storage.clear()

    def __len__(self):
        """Returns the number of files present in the cache and the underlying storage.

        Returns:
            int: the number of files present inside the root.
        """
        return len(self._list_keys())

    def __iter__(self):
        """Generator function that iterates over the keys of the cache and the underlying storage.

        Yields:
            str: the path of the object that it is iterating over, relative to the root of the provider.
        """
        yield from self._list_keys()

    def _forward(self, path, remove=False):
        """Forward the value at a given path to the next storage, and un-marks its key.
        If the value at the path is Cachable, it will only be un-dirtied if remove=True.
        """
        self._forward_value(path, self.cache_storage[path], remove)

    def _forward_value(self, path, value, remove=False):
        """Forwards a path-value pair to the next storage, and un-marks its key.

        Args:
            path (str): the path to the object relative to the root of the provider.
            value (bytes, Cachable): the value to send to the next storage.
            remove (bool, optional): cachable values are not un-marked automatically,
                as they are externally mutable. Set this to True to un-mark them anyway.
        """
        cachable = isinstance(value, Cachable)

        if not cachable or remove:
            self.dirty_keys.discard(path)

        if cachable:
            self.next_storage[path] = value.tobytes()
        else:
            self.next_storage[path] = value

    def _free_up_space(self, extra_size: int):
        """Helper function that frees up space the requred space in cache.
            No action is taken if there is sufficient space in the cache.

        Args:
            extra_size (int): the space that needs is required in bytes.
        """
        while self.cache_used > 0 and extra_size + self.cache_used > self.cache_size:
            self._pop_from_cache()

    def _pop_from_cache(self):
        """Helper function that pops the least recently used key, value pair from the cache"""
        key, itemsize = self.lru_sizes.popitem(last=False)
        if key in self.dirty_keys:
            self._forward(key, remove=True)
        del self.cache_storage[key]
        self.cache_used -= itemsize

    def _insert_in_cache(self, path: str, value: Union[bytes, Cachable]):
        """Helper function that adds a key value pair to the cache.

        Args:
            path (str): the path relative to the root of the underlying storage.
            value (bytes): the value to be assigned at the path.

        Raises:
            ReadOnlyError: If the provider is in read-only mode.
        """
        self.check_readonly()
        self._free_up_space(len(value))
        self.cache_storage[path] = value  # type: ignore
        self.cache_used += len(value)
        self.lru_sizes[path] = len(value)

    def _list_keys(self):
        """Helper function that lists all the objects present in the cache and the underlying storage.

        Returns:
            list: list of all the objects found in the cache and the underlying storage.
        """
        all_keys = {key for key in self.next_storage}
        for key in self.cache_storage:
            all_keys.add(key)
        return list(all_keys)
