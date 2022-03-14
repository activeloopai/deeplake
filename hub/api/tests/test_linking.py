import pytest
import hub
import numpy as np


def test_linking(memory_ds):
    ds = memory_ds
    ds.create_tensor("x")
    ds.create_tensor("y")
    ds._link_tensors("x", "y", "append_test")
    ds.x.extend(list(range(10)))
    np.testing.assert_array_equal(ds.x.numpy(), np.arange(10).reshape(-1, 1))
    np.testing.assert_array_equal(ds.x.numpy(), ds.y.numpy())
