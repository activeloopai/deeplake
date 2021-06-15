"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import os
import sys
from timeit import default_timer
import psutil

import hub_v1
from hub_v1 import config
from hub_v1.exceptions import ModuleNotInstalledException

try:
    from dask.cache import Cache
except ImportError:
    Cache = object

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
    if "dask" not in sys.modules:
        raise ModuleNotInstalledException("dask")
    else:
        import dask
        from dask.distributed import Client

        global dask
        global Client

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

        local_directory = os.path.join(
            os.path.expanduser("~"),
            ".activeloop",
            "tmp",
        )
        if not os.path.exists(local_directory):
            os.makedirs(local_directory)
        client = Client(
            n_workers=n_workers,
            processes=processes,
            memory_limit=memory_limit,
            threads_per_worker=threads_per_worker,
            local_directory=local_directory,
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


if "dask.cache" in sys.modules:
    cache = HubCache(2e8)
    cache.register()
