from multiprocessing import Manager
from hub.core.compute.provider import ComputeProvider, SharedValue
from pathos.pools import ThreadPool  # type: ignore
import ctypes


class ThreadProvider(ComputeProvider):
    def __init__(self, workers):
        self.workers = workers
        self._manager = Manager()
        self.pool = ThreadPool(nodes=workers)

    def map(self, func, iterable):
        return self.pool.map(func, iterable)

    def create_shared_value(self) -> SharedValue:
        return ManagedValue(self._manager)

    def close(self):
        self.pool.close()
        self.pool.join()
        self.pool.clear()


class ManagedValue(SharedValue):
    def __init__(self, manager) -> None:
        super().__init__()
        self._val = manager.Value(ctypes.c_uint64, 0)

    def set(self, val) -> None:
        self._val.value = val

    def get(self):
        return self._val.value
