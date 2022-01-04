import pytest
from hub.util.check_installation import ray_installed
from hub.util.compute import get_compute_provider

schedulers = ["threaded", "processed", "serial"]
schedulers = schedulers + ["ray"] if ray_installed() else schedulers
all_schedulers = pytest.mark.parametrize("scheduler", schedulers)


@all_schedulers
def test_compute_with_progress_bar(scheduler):
    def f(pg_callback, x):
        pg_callback(1)
        return x * 2

    compute = get_compute_provider(scheduler=scheduler, num_workers=2)
    try:
        r = compute.map_with_progressbar(f, range(1000), 1000)

        assert r is not None
        assert len(r) == 1000

    finally:
        compute.close()
