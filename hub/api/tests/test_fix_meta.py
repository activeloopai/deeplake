import pytest
import os
import numpy as np
import json

import hub
from hub.constants import KB
from hub.util.exceptions import CorruptedMetaError


def assert_correct(ds):
    assert ds.x.num_chunks == 8
    assert ds.x.shape_interval.lower == (1001, 10, 9)
    assert ds.x.shape_interval.upper == (1001, 11, 10)
    assert ds.y.shape_interval.lower == (101, 10, 10)
    assert ds.y.shape_interval.upper == (101, 11, 11)

    np.testing.assert_array_equal(
        ds.x[0:1000].numpy(), np.ones((1000, 10, 10), dtype="int32")
    )
    np.testing.assert_array_equal(ds.x[1000].numpy(), np.ones((11, 9), dtype="int32"))


def populate(ds):
    ds.create_tensor(
        "x", dtype="int32", sample_compression=None, max_chunk_size=100 * KB
    )
    ds.create_tensor(
        "y", dtype="int32", sample_compression=None, max_chunk_size=100 * KB
    )
    ds.y.extend(np.ones((100, 11, 11), dtype="int32"))
    ds.y.append(np.ones((10, 10), dtype="int32"))

    ds.x.extend(np.ones((1000, 10, 10), dtype="int32"))
    ds.x.append(np.ones((11, 9), dtype="int32"))
    assert_correct(ds)


def corrupt(ds, tensor_name: str):
    x_dir = os.path.join(ds.path, tensor_name)
    x_meta_path = os.path.join(x_dir, "tensor_meta.json")

    with open(x_meta_path, "r") as f:
        meta = json.load(f)
        meta["length"] = 0
        meta["min_shape"] = [0]
        meta["max_shape"] = [0]

    with open(x_meta_path, "w") as f:
        json.dump(meta, f)


def test_wrong_tensor_meta_length(local_ds_generator):
    """If a tensor_meta.json's length was incorrectly set."""

    ds = local_ds_generator()
    populate(ds)

    ds = local_ds_generator()
    corrupt(ds, "x")

    with pytest.raises(CorruptedMetaError):
        ds = local_ds_generator()

    with pytest.raises(ValueError):
        # require `backed_up=True`
        hub.fix(ds.path)

    hub.fix(ds.path, backed_up=True)

    ds = local_ds_generator()

    assert_correct(ds)

    # re-corrupt and make sure fixing isn't happening automatically
    corrupt(ds, "x")
    with pytest.raises(CorruptedMetaError):
        ds = local_ds_generator()
