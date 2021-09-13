from hub.core.compute.provider import ComputeProvider
from pathos.pools import ProcessPool  # type: ignore


class ProcessProvider(ComputeProvider):
    def __init__(self, workers):
        self.workers = workers
        self.pool = ProcessPool(nodes=workers)

    def map(self, func, iterable):
        return self.pool.map(func, iterable)

    def close(self):
        self.pool.close()
        self.pool.join()
        self.pool.clear()
