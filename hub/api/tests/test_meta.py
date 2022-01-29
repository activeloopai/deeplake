from numpy import ones as np_ones

from hub import __version__
from hub.api.tests.test_api import MAX_FLOAT_DTYPE


def test_version(local_ds_generator):
    ds = local_ds_generator()
    assert ds.meta.version == __version__

    # persistence
    ds = local_ds_generator()
    assert ds.meta.version == __version__


def test_subsequent_updates(local_ds_generator):
    with local_ds_generator() as ds:
        ds.create_tensor("tensor")  # dtype is not specified

    assert ds.tensor.dtype == None

    with local_ds_generator() as ds:
        ds.tensor.extend(np_ones((5, 100, 100)))

    # dtype is auto-specified
    assert ds.tensor.dtype == MAX_FLOAT_DTYPE

    ds = local_ds_generator()
    assert len(ds) == 5

    with local_ds_generator() as ds:
        for _ in range(5):
            ds.tensor.append(np_ones((100, 100)))

    ds = local_ds_generator()
    assert len(ds) == 10
    assert ds.tensor.shape == (10, 100, 100)

    ds = local_ds_generator()

    with local_ds_generator() as ds:
        for _ in range(5):
            ds.tensor.append(np_ones((100, 200)))

    assert ds.tensor.shape == (15, 100, None)
    si = ds.tensor.shape_interval
    assert si.lower == (15, 100, 100)
    assert si.upper == (15, 100, 200)
