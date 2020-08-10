import psutil

import dask
import hub
from dask.cache import Cache
from dask.distributed import Client
from hub import config

_client = None

cache = Cache(2e9)
cache.register()


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
