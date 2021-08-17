from typing import Tuple, Dict
from hub.util.exceptions import LockedException
from hub.util.path import get_path_from_storage
from hub.util.threading import terminate_thread
from hub.core.storage.provider import StorageProvider
import hub
import time
import struct
import atexit
import uuid
import os
import threading


class Lock(object):
    """Locks a StorageProvider instance to avoid concurrent writes from multiple machines.

    Example:
        From machine 1:
        s3 = hub.core.storage.S3Provider(S3_URL)
        lock = hub.core.lock.Lock(s3)  # Works

        From machine 2:
        s3 = hub.core.storage.S3Provider(S3_URL)
        lock = hub.core.lock.lock(s3)  # Raises LockedException

        The lock is updated every 2 mins by an internal thread. The lock is valid for 5 mins after the last update.

    Args:
        storage (StorageProvider): The storage provder to be locked.

    Raises:
        LockedException: If the storage is already locked by a different machine.
    """

    def __init__(self, storage: StorageProvider):
        self.storage = storage
        self.acquired = False
        self._thread_lock = threading.Lock()
        self.acquire()
        atexit.register(self.release)

    def _get_lock_bytes(self) -> bytes:
        return uuid.getnode().to_bytes(6, "little") + struct.pack("d", time.time())

    def _parse_lock_bytes(self, byts) -> Tuple[int, float]:
        byts = memoryview(byts)
        nodeid = int.from_bytes(byts[:6], "little")
        timestamp = struct.unpack("d", byts[6:])[0]
        return nodeid, timestamp

    def _lock_loop(self):
        try:
            while True:
                try:
                    self.storage[
                        hub.constants.DATASET_LOCK_FILENAME
                    ] = self._get_lock_bytes()
                except Exception:
                    pass
                time.sleep(hub.constants.DATASET_LOCK_UPDATE_INTERVAL)
        except Exception:  # Thread termination
            return

    def acquire(self):
        if self.acquired:
            return
        self.storage.check_readonly()
        lock_bytes = self.storage.get(hub.constants.DATASET_LOCK_FILENAME)
        if lock_bytes is not None:
            nodeid = None
            try:
                nodeid, timestamp = self._parse_lock_bytes(lock_bytes)
            except Exception:  # parse error from corrupt lock file, ignore
                pass
            if nodeid:
                if nodeid == uuid.getnode():
                    # Lock left by this machine from a previous run, ignore
                    pass
                elif time.time() - timestamp < hub.constants.DATASET_LOCK_VALIDITY:
                    raise LockedException()

        self._thread = threading.Thread(target=self._lock_loop, daemon=True)
        self._thread.start()
        self.acquired = True

    def release(self):
        if not self.acquired:
            return
        with self._thread_lock:
            terminate_thread(self._thread)
            self._acquired = False


_LOCKS: Dict[str, Lock] = {}


def lock(storage: StorageProvider):
    """Locks a StorageProvider instance to avoid concurrent writes from multiple machines.

    Args:
        storage (StorageProvider): The storage provder to be locked.

    Raises:
        LockedException: If the storage is already locked by a different machine.
    """

    path = get_path_from_storage(storage)
    lock = _LOCKS.get(path)
    if lock:
        lock.acquire()
    else:
        lock = Lock(storage)
        _LOCKS[path] = lock


def unlock(storage: StorageProvider):
    """Unlocks a storage provider that was locked by this machine

    Args:
        storage (StorageProvider): The storage provder to be locked.
    """
    path = get_path_from_storage(storage)
    lock = _LOCKS.get(path)
    if lock:
        lock.release()
