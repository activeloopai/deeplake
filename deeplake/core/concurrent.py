from deeplake.core.lock import Lock
from deeplake.constants import VERSION_CONTROL_INFO_LOCK_FILENAME
import uuid

class ConcurrentDatasetWriter:

    def __init__(self, ds):
        if ds.commit_id is None:
            raise ValueError("Dataset must be committed from master process before writing to it concurrently.")
        ds.checkout(ds.commit_id)
        self.ds = ds
        self.original_branch = ds.branch
        self.concurrent_branch = f"concurrent_{uuid.uuid4().hex[:8]}"

    def __enter__(self):
        ds = self.ds
        ds.read_only = False
        ds.checkout(self.concurrent_branch, create=True)

    def __exit__(self, *_, **__):
        ds = self.ds
        lock = Lock(ds.base_storage, VERSION_CONTROL_INFO_LOCK_FILENAME)
        lock.acquire()
        ds._locking_enabled = False
        try:
            ds.checkout(self.original_branch)
            ds.merge(self.concurrent_branch)
        finally:
            ds._locking_enabled = True
            lock.release()

    def __getitem__(self, key: str):
        return self.ds[key]

    def __getattribute__(self, attr: str):
        return getattr(self.ds, attr)
