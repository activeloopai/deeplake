from uuid import uuid4
from deeplake.core.compute.provider import ComputeProvider
from pathos.helpers import mp as pathos_multiprocess  # type: ignore
from pathos.pools import ThreadPool  # type: ignore


class ThreadProvider(ComputeProvider):
    def __init__(self, workers):
        self.workers = workers
        self.manager = pathos_multiprocess.Manager()
        self.pool = ThreadPool(nodes=workers, id=uuid4())

    def map(self, func, iterable):
        return self.pool.map(func, iterable)

    def create_queue(self):
        return self.manager.Queue()

    def close(self):
        self.pool.close()
        self.pool.join()
        self.pool.clear()

        if self.manager is not None:
            self.manager.shutdown()
            self.manager = None
