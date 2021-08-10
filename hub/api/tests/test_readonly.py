from typing import Tuple
import pytest
import numpy as np

import hub
from hub.util.exceptions import CouldNotCreateNewDatasetException, ReadOnlyModeError


def _assert_readonly_ops(ds, num_samples: int, sample_shape: Tuple[int]):
    assert ds.read_only

    with pytest.raises(ReadOnlyModeError):
        ds.tensor.append(np.ones(sample_shape))

    with pytest.raises(ReadOnlyModeError):
        ds.tensor[0] = np.ones((200, 200))

    assert len(ds) == num_samples
    assert len(ds.tensor) == num_samples

    assert ds.tensor.shape == (num_samples, *sample_shape)


def test_readonly(local_ds_generator):
    ds = local_ds_generator()
    ds.create_tensor("tensor")
    ds.tensor.append(np.ones((100, 100)))
    ds.read_only = True
    _assert_readonly_ops(ds, 1, (100, 100))

    ds = local_ds_generator()
    ds.read_only = True
    _assert_readonly_ops(ds, 1, (100, 100))

    with pytest.raises(ReadOnlyModeError):
        ds.info.update(key=0)

    with pytest.raises(ReadOnlyModeError):
        ds.tensor.info.update(key=0)


@pytest.mark.xfail(raises=CouldNotCreateNewDatasetException, strict=True)
def test_readonly_doesnt_exist(local_path):
    hub.dataset(local_path, read_only=True)
