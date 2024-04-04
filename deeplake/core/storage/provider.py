from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from typing import Optional, Set, Sequence, Dict, final

from deeplake.constants import BYTE_PADDING
from deeplake.util.assert_byte_indexes import assert_byte_indexes
from deeplake.util.exceptions import ReadOnlyModeError
from deeplake.util.keys import get_dataset_lock_key
import posixpath
import threading

_STORAGES: Dict[str, "StorageProvider"] = {}


def storage_factory(cls, root: str = "", *args, **kwargs) -> "StorageProvider":
    if cls.__name__ == "MemoryProvider":
        return cls(root, *args, **kwargs)
    thread_id = threading.get_ident()
    try:
        return _STORAGES[f"{thread_id}_{root}_{args}_{kwargs}"]
    except KeyError:
        storage = cls(root, *args, **kwargs)
        _STORAGES[f"{thread_id}_{root}_{args}_{kwargs}"] = storage
        return storage


class StorageProvider(ABC, MutableMapping):
    autoflush = False
    read_only = False
    root = ""
    _is_hub_path = False

    """An abstract base class for implementing a storage provider.

    To add a new provider using Provider, create a subclass and implement all 5 abstract methods below.
    """

    def __init__(self):
        self._temp_data: dict[str, bytes] = {}

    def _is_temp(self, key: str) -> bool:
        """Check if the key is a temporary key and shouldn't be persisted to storage"""
        return key.startswith("__temp")

    @final
    def __getitem__(self, path: str):
        """Gets the object present at the path within the given byte range.

        Example:

            >>> my_data = my_provider["abc.txt"]

        Args:
            path (str): The path relative to the root of the provider.

        Returns:
            bytes: The bytes of the object present at the path.

        Raises:
            KeyError: If an object is not found at the path.
            DirectoryAtPathException: If a directory is found at the path.
            Exception: Any other exception encountered while trying to fetch the object.
        """
        if self._is_temp(path):
            return self._temp_data[path]

        return self._getitem_impl(path)

    @abstractmethod
    def _getitem_impl(self, path: str):
        """Gets the object present at the path within the given byte range.

        Args:
            path (str): The path relative to the root of the provider.

        Returns:
            bytes: The bytes of the object present at the path.

        Raises:
            KeyError: If an object is not found at the path.
        """

    @final
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
        if self._is_temp(path):
            return self._temp_data[path]

        return self._get_bytes_impl(path, start_byte, end_byte)

    def _get_bytes_impl(
        self,
        path: str,
        start_byte: Optional[int] = None,
        end_byte: Optional[int] = None,
    ):
        assert_byte_indexes(start_byte, end_byte)
        return self[path][start_byte:end_byte]

    def __setitem__(self, path: str, value: bytes):
        """Sets the object present at the path with the value

        Example:

            >>> my_provider["abc.txt"] = b"abcd"

        Args:
            path (str): the path relative to the root of the provider.
            value (bytes): the value to be assigned at the path.

        Raises:
            Exception: If unable to set item due to directory at path or permission or space issues.
            FileAtPathException: If the directory to the path is a file instead of a directory.
            ReadOnlyError: If the provider is in read-only mode.
        """
        # print(f"Setitem: {path} in {self}")
        self.check_readonly()
        if self._is_temp(path):
            self._temp_data[path] = value
            return

        self._setitem_impl(path, value)

    @abstractmethod
    def _setitem_impl(self, path: str, value: bytes):
        """Sets the object present at the path with the value

        Args:
            path (str): the path relative to the root of the provider.
            value (bytes): the value to be assigned at the path.
        """

    def set_bytes(
        self,
        path: str,
        value: bytes,
        start_byte: Optional[int] = None,
        overwrite: Optional[bool] = False,
    ):
        """Sets the object present at the path with the value

        Args:
            path (str): the path relative to the root of the provider.
            value (bytes): the value to be assigned at the path.
            start_byte (int, optional): If only specific bytes starting from start_byte are to be assigned.
            overwrite (boolean, optional): If the value is True, if there is an object present at the path
                it is completely overwritten, without fetching it's data.

        Raises:
            InvalidBytesRequestedError: If `start_byte` < 0.
            ReadOnlyModeError: If the provider is in read-only mode.
        """
        self.check_readonly()
        start_byte = start_byte or 0
        end_byte = start_byte + len(value)
        assert_byte_indexes(start_byte, end_byte)

        if path in self and not overwrite:
            current_value = bytearray(self[path])
            # need to pad with zeros at the end to write extra bytes
            if end_byte > len(current_value):
                current_value = current_value.ljust(end_byte, BYTE_PADDING)
            current_value[start_byte:end_byte] = value
            self[path] = current_value
        else:
            # need to pad with zeros at the start to write from an offset
            if start_byte != 0:
                value = value.rjust(end_byte, BYTE_PADDING)
            self[path] = value

    @final
    def __contains__(self, key) -> bool:
        if self._is_temp(key):
            return key in self._temp_data

        return self._contains_impl(key)

    def _contains_impl(self, key) -> bool:
        """Check if the key exists in the provider."""
        return key in self._all_keys_impl()

    @final
    def __iter__(self):
        """Generator function that iterates over the keys of the provider.

        Example:

            >>> for my_data in my_provider:
            ...    pass

        Yields:
            str: the path of the object that it is iterating over, relative to the root of the provider.
        """
        yield from self._all_keys()

    @abstractmethod
    def _all_keys_impl(self, refresh: bool = False) -> Set[str]:
        pass

    @final
    def _all_keys(self, refresh: bool = False) -> Set[str]:
        """Lists all the objects present at the root of the Provider.

        Args:
            refresh (bool): refresh keys

        Returns:
            set: set of all the objects found at the root of the Provider.
        """
        return set.union(set(self._all_keys_impl(refresh)), set(self._temp_data.keys()))

    @final
    def __delitem__(self, path: str):
        """Delete the object present at the path.

        Example:

            >>> del my_provider["abc.txt"]

        Args:
            path (str): the path to the object relative to the root of the provider.

        Raises:
            KeyError: If an object is not found at the path.
            DirectoryAtPathException: If a directory is found at the path.
            Exception: Any other exception encountered while trying to fetch the object.
            ReadOnlyError: If the provider is in read-only mode.
        """
        self.check_readonly()

        if self._is_temp(path):
            del self._temp_data[path]
            return
        self._delitem_impl(path)

    @abstractmethod
    def _delitem_impl(self, path: str):
        """Delete the object present at the path."""

    @final
    def __len__(self):
        """Returns the number of files present inside the root of the provider.

        Returns:
            int: the number of files present inside the root.
        """
        return len(self._all_keys()) + len(self._temp_data)

    def enable_readonly(self):
        """Enables read-only mode for the provider."""
        self.read_only = True

    def disable_readonly(self):
        """Disables read-only mode for the provider."""
        self.read_only = False

    def check_readonly(self):
        """Raises an exception if the provider is in read-only mode."""
        if self.read_only:
            raise ReadOnlyModeError()

    def flush(self):
        """Only needs to be implemented for caches. Flushes the data to the next storage provider.
        Should be a no op for Base Storage Providers like local, s3, azure, gcs, etc.
        """
        self.check_readonly()

    def maybe_flush(self):
        """Flush cache if autoflush has been enabled.
        Called at the end of methods which write data, to ensure consistency as a default.
        """
        if self.autoflush:
            self.flush()

    @final
    def clear(self, prefix=""):
        """Delete the contents of the provider."""
        self.check_readonly()

        if prefix:
            self._temp_data = {
                k: v for k, v in self._temp_data.items() if not k.startswith(prefix)
            }
        else:
            self._temp_data = {}

        self._clear_impl(prefix)

    @abstractmethod
    def _clear_impl(self, prefix=""):
        """Delete the contents of the provider."""

    def delete_multiple(self, paths: Sequence[str]):
        for path in paths:
            del self[path]

    def empty(self) -> bool:
        lock_key = get_dataset_lock_key()
        return len(self) - int(lock_key in self) <= 0

    def copy(self):
        """Returns a copy of the provider.

        Returns:
            StorageProvider: A copy of the provider.
        """
        cls = self.__class__
        new_provider = cls.__new__(cls)
        new_provider.__setstate__(self.__getstate__())
        return new_provider

    def get_presigned_url(self, key: str) -> str:
        return posixpath.join(self.root, key)

    def get_object_size(self, key: str) -> int:
        raise NotImplementedError()

    def async_supported(self) -> bool:
        return False

    def get_items(self, keys):
        for key in keys:
            try:
                yield key, self[key]
            except KeyError:
                yield key, KeyError(key)
