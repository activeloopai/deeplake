import psutil

import dask
import hub
from dask.cache import Cache

from dask.distributed import Client
from hub import config
from multiprocessing import current_process

from dask.callbacks import Callback
from timeit import default_timer
from numbers import Number
import sys

import psutil, os, time

_client = None


def get_client():
    global _client
    if _client is None:
        _client = init()
    return _client


def init(
    token: str = "",
    cloud=False,
    n_workers=1,
    memory_limit=None,
    processes=False,
    threads_per_worker=1,
    distributed=True,
):
    """Initializes cluster either local or on the cloud
        
        Parameters
        ----------
        token: str
            token provided by snark
        cache: float
            Amount on local memory to cache locally, default 2e9 (2GB)
        cloud: bool
            Should be run locally or on the cloud
        n_workers: int
            number of concurrent workers, default to1
        threads_per_worker: int
            Number of threads per each worker
    """
    print("initialized")
    global _client
    if _client is not None:
        _client.close()

    if cloud:
        raise NotImplementedError
    elif not distributed:
        client = None
        dask.config.set(scheduler="threading")
        hub.config.DISTRIBUTED = False
    else:
        n_workers = n_workers if n_workers is not None else psutil.cpu_count()
        memory_limit = (
            memory_limit
            if memory_limit is not None
            else psutil.virtual_memory().available
        )
        client = Client(
            n_workers=n_workers,
            processes=processes,
            memory_limit=memory_limit,
            threads_per_worker=threads_per_worker,
            local_directory="/tmp/",
        )
        config.DISTRIBUTED = True

    _client = client
    return client


overhead = sys.getsizeof(1.23) * 4 + sys.getsizeof(()) * 4


class HubCache(Cache):
    def _posttask(self, key, value, dsk, state, id):
        duration = default_timer() - self.starttimes[key]
        deps = state["dependencies"][key]
        if deps:
            duration += max(self.durations.get(k, 0) for k in deps)
        self.durations[key] = duration
        nb = self._nbytes(value) + overhead + sys.getsizeof(key) * 4

        # _cost calculation has been fixed to avoid memory leak
        _cost = duration
        self.cache.put(key, value, cost=_cost, nbytes=nb)


cache = HubCache(2e9)
cache.register()
