from hub.util.exceptions import UnsupportedSchedulerError
from hub.core.compute import (
    ThreadProvider,
    ProcessProvider,
    ComputeProvider,
    SerialProvider,
)


def get_compute_provider(
    scheduler: str = "threaded", num_workers: int = 0
) -> ComputeProvider:
    num_workers = max(num_workers, 0)
    if scheduler == "serial" or num_workers == 0:
        compute: ComputeProvider = SerialProvider()
    elif scheduler == "threaded":
        compute = ThreadProvider(num_workers)
    elif scheduler == "processed":
        compute = ProcessProvider(num_workers)
    else:
        raise UnsupportedSchedulerError(scheduler)
    return compute
