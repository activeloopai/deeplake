from hub.util.exceptions import UnsupportedSchedulerError
from hub.core.compute import ThreadProvider, ProcessProvider, ComputeProvider


def get_compute_provider(
    scheduler: str = "threaded", workers: int = 1
) -> ComputeProvider:
    workers = max(workers, 1)
    if scheduler == "threaded":
        compute: ComputeProvider = ThreadProvider(workers)
    elif scheduler == "processed":
        compute = ProcessProvider(workers)
    else:
        raise UnsupportedSchedulerError(scheduler)
    return compute
