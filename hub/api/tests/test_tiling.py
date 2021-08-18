from hub.util.exceptions import CannotInferTilesError
from hub.tests.common import MAX_INT_DTYPE
import pytest
from hub.tests.dataset_fixtures import enabled_persistent_dataset_generators
import numpy as np


@enabled_persistent_dataset_generators
@pytest.mark.parametrize("compression", [None, "png"])
def test_initialize_large_samples(ds_generator, compression):
    ds = ds_generator()
    ds.create_tensor("tensor", dtype=MAX_INT_DTYPE, sample_compression=compression)
    ds.tensor.append_empty((10, 10, 3))  # small
    ds.tensor.append_empty((1000, 1000, 3))  # large
    ds.tensor.append(np.ones((10, 10, 3), dtype=MAX_INT_DTYPE))  # small
    ds.tensor.extend_empty((5, 10, 10, 3))  # small

    ds = ds_generator()
    assert ds.tensor.shape == (8, None, None, 3)
    np.testing.assert_array_equal(ds.tensor[0].numpy(), np.zeros((10, 10, 3)))
    np.testing.assert_array_equal(
        ds.tensor[1, 50:100, 50:100, :].numpy(), np.zeros((50, 50, 3))
    )
    np.testing.assert_array_equal(
        ds.tensor[1, -100:-50, -100:-50, :].numpy(), np.zeros((50, 50, 3))
    )
    np.testing.assert_array_equal(
        ds.tensor[1, -100:-50, -100:-50, :].numpy(), np.zeros((50, 50, 3))
    )

    # update large sample
    ds = ds_generator()
    ds.tensor[1, 50:100, 50:100, 0] = np.ones((50, 50, 1), dtype=MAX_INT_DTYPE)
    ds.tensor[1, 50:100, 50:100, 1] = np.ones((50, 50, 1), dtype=MAX_INT_DTYPE) * 2
    ds.tensor[1, 50:100, 50:100, 2] = np.ones((50, 50, 1), dtype=MAX_INT_DTYPE) * 3

    ds = ds_generator()
    expected = np.ones((50, 50, 3), dtype=MAX_INT_DTYPE)
    expected[:, :, 1] *= 2
    expected[:, :, 2] *= 3
    np.testing.assert_array_equal(ds.tensor[1, 50:100, 50:100, :].numpy(), expected)


def test_failures(memory_ds):
    memory_ds.create_tensor("tensor")

    with pytest.raises(CannotInferTilesError):
        # dtype must be pre-defined before an empty sample can be created (otherwise we can't infer the num chunks)
        memory_ds.tensor.append_empty((10000, 10000))
    assert memory_ds.tensor.shape == (0,)

    # fix
    memory_ds.tensor.set_dtype("uint8")
    memory_ds.tensor.append_empty((10000, 10000))
    assert memory_ds.tensor.shape == (1, 10000, 10000)
    assert memory_ds.tensor[0:5, 0:5].numpy().dtype == np.dtype("uint8")