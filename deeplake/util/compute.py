from deeplake.util.exceptions import (
    ModuleNotInstalledException,
    UnsupportedSchedulerError,
)
from deeplake.core.compute.provider import ComputeProvider


def get_compute_provider(
    scheduler: str = "threaded", num_workers: int = 0
) -> ComputeProvider:
    num_workers = max(num_workers, 0)
    if scheduler == "serial" or num_workers == 0:
        from deeplake.core.compute.serial import SerialProvider

        compute: ComputeProvider = SerialProvider()
    elif scheduler == "threaded":
        from deeplake.core.compute.thread import ThreadProvider

        compute = ThreadProvider(num_workers)
    elif scheduler == "processed":
        from deeplake.core.compute.process import ProcessProvider

        compute = ProcessProvider(num_workers)
    else:
        raise UnsupportedSchedulerError(scheduler)
    return compute
