from ray import (
    init as ray_init,
    is_initialized as ray_is_initialized
)
from ray.util.queue import Queue
from ray.util.multiprocessing import Pool

from hub.core.compute.provider import ComputeProvider


class RayProvider(ComputeProvider):
    def __init__(self, workers):
        super().__init__(workers)

        if not ray_is_initialized():
            ray_init()
        self.workers = workers
        self.pool = Pool(processes=workers)

    def map(self, func, iterable):
        return self.pool.map(func, iterable)

    def close(self):
        self.pool.close()
        self.pool.join()

    def create_queue(self):
        return Queue()
