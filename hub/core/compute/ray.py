import ray
from ray.util.multiprocessing import Pool
from hub.core.compute.provider import ComputeProvider
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

    def manager(self):
        return self._manager

    def close(self):
        self.pool.close()
        self.pool.join()
