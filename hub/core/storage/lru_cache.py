import sys
from collections import OrderedDict
from hub.constants import KB
from hub.core.partial_reader import PartialReader
from hub.core.storage.hub_memory_object import HubMemoryObject
from hub.core.chunk.base_chunk import BaseChunk
from typing import Any, Dict, Optional, Union

from hub.core.storage.provider import StorageProvider


def _get_nbytes(obj: Union[bytes, memoryview, HubMemoryObject]):
    if isinstance(obj, HubMemoryObject):
        return obj.nbytes
    return len(obj)


class LRUCache(StorageProvider):
    """LRU Cache that uses StorageProvider for caching"""

    def __init__(
        self,
        cache_storage: StorageProvider,
        next_storage: Optional[StorageProvider],
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

        self.dirty_keys: Dict[str, None] = (
            OrderedDict() if sys.version_info < (3, 7) else {}  # type: ignore
        )  # keys present in cache but not next_storage. Using a dict instead of set to preserve order.

        self.cache_used = 0
        self.hub_objects: Dict[str, HubMemoryObject] = {}

    def register_hub_object(self, path: str, obj: HubMemoryObject):
        """Registers a new object in the cache."""
        self.hub_objects[path] = obj

    def clear_hub_objects(self):
        """Removes all HubMemoryObjects from the cache."""
        self.hub_objects.clear()

    def remove_hub_object(self, path: str):
        """Removes a HubMemoryObject from the cache."""
        self.hub_objects.pop(path, None)

    def update_used_cache_for_path(self, path: str, new_size: int):
        if new_size < 0:
            raise ValueError(f"`new_size` must be >= 0. Got: {new_size}")
        if path in self.lru_sizes:
            old_size = self.lru_sizes[path]
            self.cache_used -= old_size
        self.cache_used += new_size
        self.lru_sizes[path] = new_size

    def flush(self):
        """Writes data from cache_storage to next_storage. Only the dirty keys are written.
        This is a cascading function and leads to data being written to the final storage in case of a chained cache.
        """
        self.check_readonly()
        initial_autoflush = self.autoflush
        self.autoflush = False
        for path, obj in self.hub_objects.items():
            if obj.is_dirty:
                self[path] = obj
                obj.is_dirty = False

        if self.dirty_keys:
            for key in self.dirty_keys.copy():
                self._forward(key)
            if self.next_storage is not None:
                self.next_storage.flush()

        self.autoflush = initial_autoflush

    def get_hub_object(
        self,
        path: str,
        expected_class,
        meta: Optional[Dict] = None,
        url=False,
        partial_bytes: int = 0,
    ):
        """If the data at `path` was stored using the output of a HubMemoryObject's `tobytes` function,
        this function will read it back into object form & keep the object in cache.

        Args:
            path (str): Path to the stored object.
            expected_class (callable): The expected subclass of `HubMemoryObject`.
            meta (dict, optional): Metadata associated with the stored object
            url (bool): Get presigned url instead of downloading chunk (only for videos)
            partial_bytes (int): Number of bytes to read from the beginning of the file. If 0, reads the whole file. Defaults to 0.

        Raises:
            ValueError: If the incorrect `expected_class` was provided.
            ValueError: If the type of the data at `path` is invalid.
            ValueError: If url is True but `expected_class` is not a subclass of BaseChunk.

        Returns:
            An instance of `expected_class` populated with the data.
        """
        if partial_bytes != 0:
            assert issubclass(expected_class, BaseChunk)
            if path in self.lru_sizes:
                return self[path]
            buff = self.get_bytes(path, 0, partial_bytes)
            obj = expected_class.frombuffer(buff, meta, partial=True)
            obj.data_bytes = PartialReader(self, path, header_offset=obj.header_bytes)
            if obj.nbytes <= self.cache_size:
                self._insert_in_cache(path, obj)
            return obj
        if url:
            from hub.util.remove_cache import get_base_storage

            item = get_base_storage(self).get_presigned_url(path).encode("utf-8")
            if issubclass(expected_class, BaseChunk):
                obj = expected_class.frombuffer(item, meta, url=True)
                return obj
            else:
                raise ValueError(
                    "Expected class should be subclass of BaseChunk when url is True."
                )
        else:
            item = self[path]

        if isinstance(item, HubMemoryObject):
            if type(item) != expected_class:
                raise ValueError(
                    f"'{path}' was expected to have the class '{expected_class.__name__}'. Instead, got: '{type(item)}'."
                )
            return item

        if isinstance(item, (bytes, memoryview)):
            obj = (
                expected_class.frombuffer(item)
                if meta is None
                else expected_class.frombuffer(item, meta)
            )

            if obj.nbytes <= self.cache_size:
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
        if path in self.hub_objects:
            if path in self.lru_sizes:
                self.lru_sizes.move_to_end(path)  # refresh position for LRU
            return self.hub_objects[path]
        elif path in self.lru_sizes:
            self.lru_sizes.move_to_end(path)  # refresh position for LRU
            return self.cache_storage[path]
        else:
            if self.next_storage is not None:
                # fetch from storage, may throw KeyError
                result = self.next_storage[path]

                if _get_nbytes(result) <= self.cache_size:  # insert in cache if it fits
                    self._insert_in_cache(path, result)
                return result
            raise KeyError(path)

    def get_bytes(
        self,
        path: str,
        start_byte: Optional[int] = None,
        end_byte: Optional[int] = None,
    ):
        """Gets the object present at the path within the given byte range.

        Args:
            path (str): The path relative to the root of the provider.
            start_byte (int, optional): If only specific bytes starting from start_byte are required.
            end_byte (int, optional): If only specific bytes up to end_byte are required.

        Returns:
            bytes: The bytes of the object present at the path within the given byte range.

        Raises:
            InvalidBytesRequestedError: If `start_byte` > `end_byte` or `start_byte` < 0 or `end_byte` < 0.
            KeyError: If an object is not found at the path.
        """
        if path in self.hub_objects:
            if path in self.lru_sizes:
                self.lru_sizes.move_to_end(path)  # refresh position for LRU
            return self.hub_objects[path].tobytes()[start_byte:end_byte]
        # if it is a partially read chunk in the cache, to get new bytes, we need to look at actual storage and not the cache
        elif path in self.lru_sizes and not (
            isinstance(self.cache_storage[path], BaseChunk)
            and self.cache_storage[path].is_partially_read_chunk
        ):
            self.lru_sizes.move_to_end(path)  # refresh position for LRU
            return self.cache_storage[path][start_byte:end_byte]
        else:
            if self.next_storage is not None:
                return self.next_storage.get_bytes(path, start_byte, end_byte)
            raise KeyError(path)

    def __setitem__(self, path: str, value: Union[bytes, HubMemoryObject]):
        """Puts the item in the cache_storage (if possible), else writes to next_storage.

        Args:
            path (str): the path relative to the root of the underlying storage.
            value (bytes): the value to be assigned at the path.

        Raises:
            ReadOnlyError: If the provider is in read-only mode.
        """
        self.check_readonly()
        if path in self.hub_objects:
            self.hub_objects[path].is_dirty = False

        if path in self.lru_sizes:
            size = self.lru_sizes.pop(path)
            self.cache_used -= size

        if _get_nbytes(value) <= self.cache_size:
            self._insert_in_cache(path, value)
            self.dirty_keys[path] = None
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

        if path in self.hub_objects:
            self.remove_hub_object(path)
            deleted_from_cache = True

        if path in self.lru_sizes:
            size = self.lru_sizes.pop(path)
            self.cache_used -= size
            del self.cache_storage[path]
            self.dirty_keys.pop(path, None)
            deleted_from_cache = True

        try:
            if self.next_storage is not None:
                del self.next_storage[path]
            else:
                raise KeyError(path)
        except KeyError:
            if not deleted_from_cache:
                raise

    def clear_cache(self):
        """Flushes the content of all the cache layers if not in read mode and and then deletes contents of all the layers of it.
        This doesn't delete data from the actual storage.
        """
        self._flush_if_not_read_only()
        self.clear_cache_without_flush()

    def clear_cache_without_flush(self):
        self.cache_used = 0
        self.lru_sizes.clear()
        self.dirty_keys.clear()
        self.cache_storage.clear()
        self.hub_objects.clear()
        if self.next_storage is not None and hasattr(self.next_storage, "clear_cache"):
            self.next_storage.clear_cache()

    def clear(self, prefix=""):
        """Deletes ALL the data from all the layers of the cache and the actual storage.
        This is an IRREVERSIBLE operation. Data once deleted can not be recovered.
        """
        self.check_readonly()
        if prefix:
            rm = [path for path in self.hub_objects if path.startswith(prefix)]
            for path in rm:
                self.remove_hub_object(path)

            rm = [path for path in self.lru_sizes if path.startswith(prefix)]
            for path in rm:
                size = self.lru_sizes.pop(path)
                self.cache_used -= size
                self.dirty_keys.pop(path, None)
        else:
            self.cache_used = 0
            self.lru_sizes.clear()
            self.dirty_keys.clear()
            self.hub_objects.clear()

        self.cache_storage.clear(prefix=prefix)
        if self.next_storage is not None:
            self.next_storage.clear(prefix=prefix)

    def __len__(self):
        """Returns the number of files present in the cache and the underlying storage.

        Returns:
            int: the number of files present inside the root.
        """
        return len(self._all_keys())

    def __iter__(self):
        """Generator function that iterates over the keys of the cache and the underlying storage.

        Yields:
            str: the path of the object that it is iterating over, relative to the root of the provider.
        """
        yield from self._all_keys()

    def _forward(self, path):
        """Forward the value at a given path to the next storage, and un-marks its key."""
        if self.next_storage is not None:
            self._forward_value(path, self.cache_storage[path])

    def _forward_value(self, path, value):
        """Forwards a path-value pair to the next storage, and un-marks its key.

        Args:
            path (str): the path to the object relative to the root of the provider.
            value (bytes, HubMemoryObject): the value to send to the next storage.
        """
        if self.next_storage is not None:
            self.dirty_keys.pop(path, None)

            if isinstance(value, HubMemoryObject):
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
            self._forward(key)
        del self.cache_storage[key]
        self.cache_used -= itemsize

    def _insert_in_cache(self, path: str, value: Union[bytes, HubMemoryObject]):
        """Helper function that adds a key value pair to the cache.

        Args:
            path (str): the path relative to the root of the underlying storage.
            value (bytes): the value to be assigned at the path.

        Raises:
            ReadOnlyError: If the provider is in read-only mode.
        """

        self._free_up_space(_get_nbytes(value))
        self.cache_storage[path] = value  # type: ignore

        self.update_used_cache_for_path(path, _get_nbytes(value))

    def _all_keys(self):
        """Helper function that lists all the objects present in the cache and the underlying storage.

        Returns:
            set: set of all the objects found in the cache and the underlying storage.
        """
        key_set = set()
        if self.next_storage is not None:
            key_set = self.next_storage._all_keys()  # type: ignore
        key_set = set().union(key_set, self.cache_storage._all_keys())
        for path, obj in self.hub_objects.items():
            if obj.is_dirty:
                key_set.add(path)
        return key_set

    def _flush_if_not_read_only(self):
        """Flushes the cache if not in read-only mode."""
        if not self.read_only:
            self.flush()

    def __getstate__(self) -> Dict[str, Any]:
        """Returns the state of the cache, for pickling"""

        # flushes the cache before pickling
        self._flush_if_not_read_only()

        return {
            "next_storage": self.next_storage,
            "cache_storage": self.cache_storage,
            "cache_size": self.cache_size,
        }

    def __setstate__(self, state: Dict[str, Any]):
        """Recreates a cache with the same configuration as the state.

        Args:
            state (dict): The state to be used to recreate the cache.

        Note:
            While restoring the cache, we reset its contents.
            In case the cache storage was local/s3 and is still accessible when unpickled (if same machine/s3 creds present respectively), the earlier cache contents are no longer accessible.
        """

        # TODO: We might want to change this behaviour in the future by having a separate file that keeps a track of the lru order for restoring the cache.
        # This would also allow the cache to persist across different different Dataset objects pointing to the same dataset.
        self.next_storage = state["next_storage"]
        self.cache_storage = state["cache_storage"]
        self.cache_size = state["cache_size"]
        self.lru_sizes = OrderedDict()
        self.dirty_keys = OrderedDict()
        self.cache_used = 0
        self.hub_objects = {}

    def get_object_size(self, key: str) -> int:
        if key in self.hub_objects:
            return self.hub_objects[key].nbytes

        try:
            return self.cache_storage.get_object_size(key)
        except KeyError:
            if self.next_storage is not None:
                return self.next_storage.get_object_size(key)
            raise
