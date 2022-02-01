import ray
from ray.util.multiprocessing import Pool
from hub.core.compute.provider import ComputeProvider, SharedValue
from multiprocessing import Manager


class RayProvider(ComputeProvider):
    def __init__(self, workers):
        super().__init__(workers)

        if not ray.is_initialized():
            ray.init()
        self.workers = workers
        self.pool = Pool(processes=workers)
        self._manager = Manager()

    def map(self, func, iterable):
        return self.pool.map(func, iterable)

    def create_shared_value(self):
        return RaySharedValue()

    def close(self):
        self.pool.close()
        self.pool.join()


@ray.remote
class RayRemoteValue(object):
    def __init__(self) -> None:
        self.val = 0

    def put(self, val):
        self.val = val

    def get(self):
        return self.val


class RaySharedValue(SharedValue):
    def __init__(self) -> None:
        super().__init__()
        self.val = RayRemoteValue.remote()

    def set(self, val):
        self.val.put.remote(val)

    def get(self):
        return ray.get(self.val.get.remote())
