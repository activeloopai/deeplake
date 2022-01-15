import warnings
from hub.core.compute.provider import ComputeProvider
from pathos.pools import ProcessPool  # type: ignore
from pathos.helpers import mp as pathos_multiprocess  # type: ignore


class ProcessProvider(ComputeProvider):
    def __init__(self, workers):
        self.workers = workers
        self.pool = ProcessPool(nodes=workers)
        self.manager = pathos_multiprocess.Manager()
        self._closed = False

    def map(self, func, iterable):
        return self.pool.map(func, iterable)

    def create_queue(self):
        return self.manager.Queue()

    def close(self):
        self.pool.close()
        self.pool.join()
        self.pool.clear()
        self._closed = True

    def __del__(self):
        if not self._closed:
            self.close()
            warnings.warn(
                "process pool thread leak. check compute provider is closed after use"
            )
