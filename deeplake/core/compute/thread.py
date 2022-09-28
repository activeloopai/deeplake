from multiprocessing import Manager
from deeplake.core.compute.provider import ComputeProvider
from pathos.pools import ThreadPool  # type: ignore


class ThreadProvider(ComputeProvider):
    def __init__(self, workers):
        self.workers = workers
        self.manager = Manager()
        self.pool = ThreadPool(nodes=workers)

    def map(self, func, iterable):
        return self.pool.map(func, iterable)

    def create_queue(self):
        return self.manager.Queue()

    def close(self):
        self.pool.close()
        self.pool.join()
        self.pool.clear()
