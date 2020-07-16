from dask.cache import Cache
from dask.distributed import Client

_client = None


def get_client():
    global _client
    if _client is None:
        _client = init()
    return _client


def init(
    token: str = "", cache=2e9, cloud=False, n_workers=4,
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
        client = Client(n_workers=n_workers, processes=True, memory_limit=50e9,)
    cache = Cache(cache)
    cache.register()
    global _client
    _client = client
    return client

