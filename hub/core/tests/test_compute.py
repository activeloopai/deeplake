import pytest

from hub.util.compute import get_compute_provider


@pytest.mark.parametrize("scheduler", ["threaded", "processed", "ray"])
def test_compute_with_progress_bar(scheduler):
    def f(pg_callback, x):
        pg_callback(1)
        return x * 2

    compute = get_compute_provider(scheduler=scheduler, num_workers=2)
    r = compute.map_with_progressbar(f, range(1000), 1000)

    assert r is not None
    assert len(r) == 1000


def test_serial_with_progress_bar():
    def f(x):
        return x * 2

    compute = get_compute_provider(scheduler="serial")
    r = compute.map_with_progressbar(f, range(1000), 1000)

    assert r is not None
    assert len(r) == 1000
