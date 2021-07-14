import numpy as np
import hub


def test_version(local_ds_generator):
    ds = local_ds_generator()
    assert ds.meta.version == hub.__version__

    # persistence
    ds = local_ds_generator()
    assert ds.meta.version == hub.__version__


def test_subsequent_updates(local_ds_generator):
    with local_ds_generator() as ds:
        ds.create_tensor("tensor")

    with local_ds_generator() as ds:
        ds.tensor.extend(np.ones((5, 100, 100)))

    ds = local_ds_generator()
    assert len(ds) == 5

    with local_ds_generator() as ds:
        for _ in range(5):
            ds.tensor.append(np.ones((100, 100)))

    ds = local_ds_generator()
    assert len(ds) == 10
