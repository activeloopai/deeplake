import psutil
from dask.cache import Cache
from dask.distributed import Client
from hub.log import logger

_client = None


def get_client():
    global _client
    if _client is None:
        _client = init()
    return _client


def init(
    token: str = "",
    cache=2e9,
    cloud=False,
    n_workers=None,
    memory_limit=None,
    processes=False,
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
            number of concurrent workers, default 5
    """
    if cloud:
        raise NotImplementedError
    else:
        n_workers = n_workers if n_workers is not None else psutil.cpu_count()
        memory_limit = (
            memory_limit
            if memory_limit is not None
            else psutil.virtual_memory().available
        )

        client = Client(
            n_workers=n_workers, processes=processes, memory_limit=memory_limit
        )

    cache = Cache(cache)
    cache.register()
    global _client
    _client = client
    return client
