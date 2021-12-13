import hub
import time
import uuid
import struct
import atexit
import threading

from typing import Tuple, Dict, Callable, Optional
from hub.util.exceptions import LockedException
from hub.util.keys import get_dataset_lock_key
from hub.util.path import get_path_from_storage
from hub.util.threading import terminate_thread
from hub.core.storage import StorageProvider
from hub.constants import FIRST_COMMIT_ID


def _get_lock_bytes() -> bytes:
    return uuid.getnode().to_bytes(6, "little") + struct.pack("d", time.time())


def _parse_lock_bytes(byts) -> Tuple[int, float]:
    byts = memoryview(byts)
    nodeid = int.from_bytes(byts[:6], "little")
    timestamp = struct.unpack("d", byts[6:])[0]
    return nodeid, timestamp


class Lock(object):
    def __init__(self, storage: StorageProvider, path: str):
        self.storage = storage
        self.path = path

    def acquire(self, timeout=10, force=False):
        if self.path not in self.storage:
            self.storage[self.path] = _get_lock_bytes()
            return
        nodeid, timestamp = _parse_lock_bytes(self.storage[self.path])
        if nodeid == uuid.getnode():
            self.storage[self.path] = _get_lock_bytes()
            return
        while self.path in self.storage:
            if time.time() - timestamp >= timeout:
                if force:
                    self.storage[self.path] = _get_lock_bytes()
                    return
                else:
                    raise LockedException()
            time.sleep(1)

    def release(self):
        try:
            del self.storage[self.path]
        except Exception:
            pass


class PersistentLock(Lock):
    """Locks a StorageProvider instance to avoid concurrent writes from multiple machines.

    Example:
        From machine 1:
        s3 = hub.core.storage.S3Provider(S3_URL)
        lock = hub.core.lock.Lock(s3)  # Works

        From machine 2:
        s3 = hub.core.storage.S3Provider(S3_URL)
        lock = hub.core.lock.Lock(s3)  # Raises LockedException

        The lock is updated every 2 mins by an internal thread. The lock is valid for 5 mins after the last update.

    Args:
        storage (StorageProvider): The storage provder to be locked.
        callback (Callable, optional): Called if the lock is lost after acquiring.

    Raises:
        LockedException: If the storage is already locked by a different machine.
    """

    def __init__(
        self,
        storage: StorageProvider,
        path: Optional[str] = None,
        callback: Optional[Callable] = None,
    ):
        self.storage = storage
        self.callback = callback
        self.acquired = False
        self._thread_lock = threading.Lock()
        self._previous_update_timestamp = None
        self.path = get_dataset_lock_key() if path is None else path
        self.acquire()
        atexit.register(self.release)

    def _lock_loop(self):
        try:
            while True:
                try:
                    if (
                        self._previous_update_timestamp is not None
                        and time.time() - self._previous_update_timestamp
                        >= hub.constants.DATASET_LOCK_VALIDITY
                    ):
                        # Its been too long since last update, another machine might have locked the storage
                        lock_bytes = self.storage.get(self.path)
                        if lock_bytes:
                            nodeid, timestamp = _parse_lock_bytes(lock_bytes)
                            if nodeid != uuid.getnode():
                                if self.callback:
                                    self.callback()
                                self.acquired = False
                                return
                    self._previous_update_timestamp = time.time()
                    self.storage[self.path] = _get_lock_bytes()
                except Exception:
                    pass
                time.sleep(hub.constants.DATASET_LOCK_UPDATE_INTERVAL)
        except Exception:  # Thread termination
            return

    def acquire(self):
        if self.acquired:
            return
        self.storage.check_readonly()
        lock_bytes = self.storage.get(self.path)
        if lock_bytes is not None:
            nodeid = None
            try:
                nodeid, timestamp = _parse_lock_bytes(lock_bytes)
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
        try:
            del self.storage[self.path]
        except Exception:
            pass


_LOCKS: Dict[str, Lock] = {}


def _get_lock_key(storage_path: str, commit_id: str):
    return storage_path + ":" + commit_id


def _get_lock_file_path(version: Optional[str] = None) -> str:
    if version in (None, FIRST_COMMIT_ID):
        return get_dataset_lock_key()
    return "versions/" + version + "/" + get_dataset_lock_key()  # type: ignore


def lock_version(
    storage: StorageProvider,
    version: str,
    callback: Optional[Callable] = None,
):
    """Locks a StorageProvider instance to avoid concurrent writes from multiple machines.

    Args:
        storage (StorageProvider): The storage provder to be locked.
        callback (Callable, Optional): Called if the lock is lost after acquiring.
        version (str): Commit id of the version to lock.

    Raises:
        LockedException: If the storage is already locked by a different machine.
    """
    key = _get_lock_key(get_path_from_storage(storage), version)
    lock = _LOCKS.get(key)
    if lock:
        lock.acquire()
    else:
        lock = PersistentLock(
            storage, path=_get_lock_file_path(version), callback=callback
        )
        _LOCKS[key] = lock


def unlock_version(storage: StorageProvider, version: str):
    """Unlocks a storage provider that was locked by this machine.

    Args:
        storage (StorageProvider): The storage provder to be locked.
        version (str): Commit id of the version to unlock.
    """
    key = _get_lock_key(get_path_from_storage(storage), version)
    lock = _LOCKS.get(key)
    if lock:
        lock.release()
