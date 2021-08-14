from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from hub.core.storage.cachable import Cachable
from typing import Optional

from hub.constants import (
    BYTE_PADDING,
    DATASET_LOCK_UPDATE_INTERVAL,
    DATASET_LOCK_VALIDITY,
)
from hub.util.assert_byte_indexes import assert_byte_indexes
from hub.util.exceptions import ReadOnlyModeError, LockedException
from hub.util.keys import get_dataset_lock_key
from hub.util.threading import terminate_thread


import time
import threading
import struct
import uuid
import atexit
import ctypes


_WRITE_KEY = str(uuid.uuid4()).replace("-", "").encode("ascii")  # 32 bytes


class StorageProvider(ABC, MutableMapping):
    autoflush = False
    _read_only = False
    _locked = False
    """An abstract base class for implementing a storage provider.

    To add a new provider using Provider, create a subclass and implement all 5 abstract methods below.
    """

    def __init__(self, read_only: bool = False):
        self._read_only = read_only
        if not read_only:
            if self._is_locked():
                self._locked = True
            else:
                self._lock()

    @abstractmethod
    def __getitem__(self, path: str):
        """Gets the object present at the path within the given byte range.

        Args:
            path (str): The path relative to the root of the provider.

        Returns:
            bytes: The bytes of the object present at the path.

        Raises:
            KeyError: If an object is not found at the path.
        """

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
        assert_byte_indexes(start_byte, end_byte)
        return self[path][start_byte:end_byte]

    @abstractmethod
    def __setitem__(self, path: str, value: bytes):
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
            ReadOnlyError: If the provider is in read-only mode.
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

    @abstractmethod
    def __iter__(self):
        """Generator function that iterates over the keys of the provider.

        Yields:
            str: the path of the object that it is iterating over, relative to the root of the provider.
        """

    @abstractmethod
    def __delitem__(self, path: str):
        """Delete the object present at the path.

        Args:
            path (str): the path to the object relative to the root of the provider.

        Raises:
            KeyError: If an object is not found at the path.
        """

    @abstractmethod
    def __len__(self):
        """Returns the number of files present inside the root of the provider.

        Returns:
            int: the number of files present inside the root.
        """

    @property
    def read_only(self):
        return self._locked or self._read_only

    @read_only.setter
    def read_only(self, value: bool):
        if not value and self._locked:
            raise LockedException()
        self._read_only = value

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

    def clear(self):
        """Delete the contents of the provider."""
        self.check_readonly()
        self._unlock()
        self._clear()
        self._lock()

    @abstractmethod
    def _clear(self):
        pass

    # def __contains__(self, path):
    #     try:
    #         self[path]
    #     except KeyError:
    #         return False
    #     return True

    def _is_locked(self):
        lock_bytes = self.get(get_dataset_lock_key())
        if lock_bytes is None:
            return False
        lock_bytes = memoryview(lock_bytes)
        write_key = lock_bytes[:32]
        if write_key == _WRITE_KEY:  # locked in the same python session
            return False
        lock_time = struct.unpack("d", lock_bytes[32:])[0]
        if time.time() - lock_time < DATASET_LOCK_VALIDITY:
            return True
        return False

    def _lock(self):
        self.check_readonly()
        self._stop_lock_thread = threading.Event()
        self._lock_thread = threading.Thread(target=self._lock_loop, daemon=True)
        self._lock_thread.start()
        atexit.register(self._unlock)

    def _unlock(self):
        if hasattr(self, "_lock_thread"):
            self._stop_lock_thread.set()
            terminate_thread(self._lock_thread)
            # self._lock_thread.join()
            try:
                del self[get_dataset_lock_key()]
            except Exception:
                pass  # TODO

    def _lock_loop(self):
        try:
            while not self._stop_lock_thread.is_set():
                try:
                    self[get_dataset_lock_key()] = _WRITE_KEY + struct.pack(
                        "d", time.time()
                    )
                except Exception:
                    pass
                time.sleep(DATASET_LOCK_UPDATE_INTERVAL)
        except Exception:  # Thread termination
            return

    def empty(self):
        return len(self) - int(get_dataset_lock_key() in self) <= 0
