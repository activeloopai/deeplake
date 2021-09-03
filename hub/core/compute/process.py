from hub.core.compute.provider import ComputeProvider
from pathos.pools import ProcessPool  # type: ignore
from contextlib import closing


class ProcessProvider(ComputeProvider):
    def __init__(self, workers):
        self.workers = workers
        self.pool = ProcessPool(nodes=workers)

    def map(self, func, iterable):
        with closing(self.pool) as p:
            res = p.map(func, iterable)
        return res
