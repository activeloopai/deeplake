from typing import Tuple
import pytest
import numpy as np

from hub import Dataset
from hub.util.exceptions import ReadOnlyModeError


def _assert_readonly_ops(ds, num_samples: int, sample_shape: Tuple[int]):
    assert ds.read_only

    with pytest.raises(ReadOnlyModeError):
        ds.tensor.append(np.ones(sample_shape))

    assert len(ds) == num_samples
    assert len(ds.tensor) == num_samples

    assert ds.tensor.shape == (num_samples, *sample_shape)


def test_readonly(local_ds):
    path = local_ds.path

    local_ds.create_tensor("tensor")
    local_ds.tensor.append(np.ones((100, 100)))
    local_ds.read_only = True
    _assert_readonly_ops(local_ds, 1, (100, 100))
    del local_ds

    local_ds = Dataset(path)
    local_ds.read_only = True
    _assert_readonly_ops(local_ds, 1, (100, 100))
