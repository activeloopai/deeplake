import ray
from ray.util.multiprocessing import Pool
from hub.core.compute.provider import ComputeProvider


class RayProvider(ComputeProvider):
    def __init__(self, workers):
        if not ray.is_initialized():
            ray.init()
        self.workers = workers
        self.pool = Pool(processes=workers)

    def map(self, func, iterable):
        return self.pool.map(func, iterable)

    def close(self):
        self.pool.close()
        self.pool.join()
