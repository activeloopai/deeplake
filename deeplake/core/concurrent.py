from deeplake.core.lock import Lock
from deeplake.constants import VERSION_CONTROL_INFO_LOCK_FILENAME
import uuid


class ConcurrentDatasetWriter:
    def __init__(self, ds):
        if ds.commit_id is None:
            raise ValueError(
                "Dataset must be committed from master process before writing to it concurrently."
            )
        ds.checkout(ds.commit_id)
        self.ds = ds
        self.original_branch = ds.branch
        self.concurrent_branch = f"concurrent_{uuid.uuid4().hex[:8]}"
        ds.read_only = False
        ds._locking_enabled = False
        ds.checkout(self.concurrent_branch, create=True)

    def __enter__(self):
        return self

    def __exit__(self, *_, **__):
        self.join()

    def __getitem__(self, key: str):
        return self.ds[key]

    def __getattribute__(self, attr: str):
        try:
            return super().__getattribute__(attr)
        except AttributeError:
            return getattr(self.ds, attr)

    def _merge(self):
        ds = self.ds
        lock = Lock(ds.base_storage, VERSION_CONTROL_INFO_LOCK_FILENAME)
        lock.acquire()
        try:
            ds.checkout(self.original_branch)
            ds.merge(self.concurrent_branch)
        finally:
            lock.release()

    def join(self):
        self._merge()
        self.ds.checkout(self.concurrent_branch)
