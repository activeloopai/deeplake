from hub.core.compute.provider import ComputeProvider
from ray.util.multiprocessing import Pool


class RayProvider(ComputeProvider):
    def __init__(self, workers):
        self.workers = workers
        self.pool = Pool(processes=workers)

    def map(self, func, iterable):
        return self.pool.map(func, iterable)

    def close(self):
        self.pool.close()
        self.pool.join()
