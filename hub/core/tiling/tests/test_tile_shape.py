from hub.core.tiling.tile import get_tile_shape
import numpy as np


def test_tile_shape():
    x = np.random.random((1000, 500, 3))
    tile_shape = get_tile_shape(
        x.shape, sample_size=x.nbytes, chunk_size=x.nbytes / 16, exclude_axes=2
    )
    assert np.prod(tile_shape) * x.itemsize == x.nbytes / 16
    assert tile_shape[2] == 3
