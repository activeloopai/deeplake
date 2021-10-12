from hub.util.exceptions import ModuleNotInstalledException, UnsupportedSchedulerError
from hub.util.check_installation import ray_installed
from hub.core.compute.provider import ComputeProvider


def get_compute_provider(
    scheduler: str = "threaded", num_workers: int = 0
) -> ComputeProvider:
    num_workers = max(num_workers, 0)
    if scheduler == "serial" or num_workers == 0:
        from hub.core.compute.serial import SerialProvider

        compute: ComputeProvider = SerialProvider()
    elif scheduler == "threaded":
        from hub.core.compute.thread import ThreadProvider

        compute = ThreadProvider(num_workers)
    elif scheduler == "processed":
        from hub.core.compute.process import ProcessProvider

        compute = ProcessProvider(num_workers)
    elif scheduler == "ray":
        if not ray_installed():
            raise ModuleNotInstalledException(
                "'ray' should be installed to use ray scheduler."
            )
        from hub.core.compute.ray import RayProvider

        compute = RayProvider(num_workers)

    else:
        raise UnsupportedSchedulerError(scheduler)
    return compute
