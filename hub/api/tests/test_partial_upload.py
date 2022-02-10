import hub
import pytest
import numpy as np


def test_partial_upload(memory_ds):
    ds = memory_ds
    ds.create_tensor("image", htype="image", sample_compression="jpeg")
    ds.image.append(hub.tiled(sample_shape=(1000, 1000, 3), tile_shape=(10, 10, 3)))
    expected = np.zeros((1, 1000, 1000, 3), dtype=np.uint8)
    np.testing.assert_array_equal(ds.image.numpy(), expected)
    r = np.random.random((217, 212, 2))
    expected[0, -217:, :212, 1:] = r
    ds.image[0][-217:, :212, 1:] = r
    np.testing.assert_array_equal(ds.image.numpy(), expected)
