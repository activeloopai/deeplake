import warnings
from hub.core.compute.provider import ComputeProvider, SharedValue
from pathos.pools import ProcessPool  # type: ignore
from pathos.helpers import mp as pathos_multiprocess  # type: ignore
import ctypes


class ProcessProvider(ComputeProvider):
    def __init__(self, workers):
        self.workers = workers
        self.pool = ProcessPool(nodes=workers)
        self._manager = pathos_multiprocess.Manager()
        self._closed = False

    def map(self, func, iterable):
        return self.pool.map(func, iterable)

    def create_shared_value(self):
        return ManagedValue(self._manager)

    def close(self):
        self.pool.close()
        self.pool.join()
        self.pool.clear()
        self._closed = True

    def __del__(self):
        if not self._closed:
            self.close()
            warnings.warn(
                "process pool thread leak. check compute provider is closed after use"
            )


class ManagedValue(SharedValue):
    def __init__(self, manager) -> None:
        super().__init__()
        self._val = manager.Value(ctypes.c_uint64, 0)

    def set(self, val) -> None:
        self._val.value = val

    def get(self):
        return self._val.value
