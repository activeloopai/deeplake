import deeplake
import time
import uuid
import struct
import atexit
import threading

from typing import Tuple, Dict, Callable, Optional, Set
from collections import defaultdict
from deeplake.util.exceptions import LockedException
from deeplake.util.keys import get_dataset_lock_key
from deeplake.util.remove_cache import get_base_storage
from deeplake.util.path import get_path_from_storage
from deeplake.util.threading import terminate_thread
from deeplake.core.storage import StorageProvider
from deeplake.constants import FIRST_COMMIT_ID
from deeplake.client.utils import get_user_name


def _get_lock_bytes(username: Optional[str] = None) -> bytes:
    byts = uuid.getnode().to_bytes(6, "little") + struct.pack("d", time.time())
    ## TODO Uncomment lines below once this version has propogated through the userbase.
    # if username:
    #     byts += username.encode("utf-8")
    return byts


def _parse_lock_bytes(byts) -> Tuple[int, int, str]:
    byts = memoryview(byts)
    nodeid = int.from_bytes(byts[:6], "little")
    timestamp = struct.unpack("d", byts[6:14])[0]
    username = str(byts[14:], "utf-8")
    return nodeid, timestamp, username


class Lock(object):
    def __init__(self, storage: StorageProvider, path: str):
        self.storage = storage
        self.path = path
        username = get_user_name()
        if username == "public":
            self.username = None
        else:
            self.username = username

    def _write_lock(self):
        storage = self.storage
        try:
            read_only = storage.read_only
            storage.disable_readonly()
            storage[self.path] = _get_lock_bytes(self.username)
        finally:
            if read_only:
                storage.enable_readonly()

    def acquire(self, timeout=10, force=False):
        storage = self.storage
        path = self.path
        try:
            nodeid, timestamp, _ = _parse_lock_bytes(storage[path])
        except KeyError:
            return self._write_lock()
        if nodeid == uuid.getnode():
            return self._write_lock()
        while path in storage:
            if time.time() - timestamp >= timeout:
                if force:
                    return self._write_lock()
                else:
                    raise LockedException()
            time.sleep(1)

    def release(self):
        storage = self.storage
        try:
            read_only = storage.read_only
            storage.disable_readonly()
            del storage[self.path]
        except Exception:
            pass
        finally:
            if read_only:
                storage.enable_readonly()

    def __enter__(self):
        self.acquire()

    def __exit__(self, *args, **kwargs):
        self.release()


class PersistentLock(Lock):
    """Locks a StorageProvider instance to avoid concurrent writes from multiple machines.

    Example:
        From machine 1:
        s3 = deeplake.core.storage.S3Provider(S3_URL)
        lock = deeplake.core.lock.Lock(s3)  # Works

        From machine 2:
        s3 = deeplake.core.storage.S3Provider(S3_URL)
        lock = deeplake.core.lock.Lock(s3)  # Raises LockedException

        The lock is updated every 2 mins by an internal thread. The lock is valid for 5 mins after the last update.

    Args:
        storage (StorageProvider): The storage provder to be locked.
        lock_lost_callback (Callable, optional): Called if the lock is lost after acquiring.

    Raises:
        LockedException: If the storage is already locked by a different machine.
    """

    def __init__(
        self,
        storage: StorageProvider,
        path: Optional[str] = None,
        lock_lost_callback: Optional[Callable] = None,
    ):
        self.storage = storage
        self.path = get_dataset_lock_key() if path is None else path
        self.lock_lost_callback = lock_lost_callback
        self.acquired = False
        self._thread_lock = threading.Lock()
        self._previous_update_timestamp = None
        username = get_user_name()
        if username == "public":
            self.username = None
        else:
            self.username = username
        self.acquire()
        atexit.register(self.release)

    def _lock_loop(self):
        try:
            while True:
                try:
                    if (
                        self._previous_update_timestamp is not None
                        and time.time() - self._previous_update_timestamp
                        >= deeplake.constants.DATASET_LOCK_VALIDITY
                    ):
                        # Its been too long since last update, another machine might have locked the storage
                        lock_bytes = self.storage.get(self.path)
                        if lock_bytes:
                            nodeid, _, _ = _parse_lock_bytes(lock_bytes)
                            if nodeid != uuid.getnode():
                                if self.lock_lost_callback:
                                    self.lock_lost_callback()
                                self.acquired = False
                                return
                        elif not self._init:
                            if self.lock_lost_callback:
                                self.lock_lost_callback()
                            self.acquired = False
                            return
                    self._previous_update_timestamp = time.time()
                    self.storage[self.path] = _get_lock_bytes(self.username)
                except Exception:
                    pass
                self._init = False
                time.sleep(deeplake.constants.DATASET_LOCK_UPDATE_INTERVAL)
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
                nodeid, timestamp, _ = _parse_lock_bytes(lock_bytes)
            except Exception:  # parse error from corrupt lock file, ignore
                pass
            if nodeid:
                if nodeid == uuid.getnode():
                    # Lock left by this machine from a previous run, ignore
                    pass
                elif time.time() - timestamp < deeplake.constants.DATASET_LOCK_VALIDITY:
                    raise LockedException()

        self._init = True
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
_REFS: Dict[str, Set[int]] = defaultdict(set)


def _get_lock_key(storage_path: str, commit_id: str):
    return storage_path + ":" + commit_id


def _get_lock_file_path(version: Optional[str] = None) -> str:
    if version in (None, FIRST_COMMIT_ID):
        return get_dataset_lock_key()
    return "versions/" + version + "/" + get_dataset_lock_key()  # type: ignore


def lock_dataset(
    dataset,
    lock_lost_callback: Optional[Callable] = None,
):
    """Locks a StorageProvider instance to avoid concurrent writes from multiple machines.

    Args:
        dataset: Dataset instance.
        lock_lost_callback (Callable, Optional): Called if the lock is lost after acquiring.

    Raises:
        LockedException: If the storage is already locked by a different machine.
    """
    storage = get_base_storage(dataset.storage)
    version = dataset.version_state["commit_id"]
    key = _get_lock_key(get_path_from_storage(storage), version)
    lock = _LOCKS.get(key)
    if lock:
        lock.acquire()
    else:
        lock = PersistentLock(
            storage,
            path=_get_lock_file_path(version),
            lock_lost_callback=lock_lost_callback,
        )
        _LOCKS[key] = lock
    _REFS[key].add(id(dataset))


def unlock_dataset(dataset):
    """Unlocks a storage provider that was locked by this machine.

    Args:
        dataset: The dataset to be unlocked
    """
    storage = get_base_storage(dataset.storage)
    version = dataset.version_state["commit_id"]
    key = _get_lock_key(get_path_from_storage(storage), version)
    try:
        lock = _LOCKS[key]
        ref_set = _REFS[key]
        try:
            ref_set.remove(id(dataset))
        except KeyError:
            pass
        if not ref_set:
            lock.release()
            del _REFS[key]
            del _LOCKS[key]
    except KeyError:
        pass
