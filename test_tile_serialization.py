import numpy as np
from serialize_tile import *
import gzip


def test_break_into_tiles():
    sample = np.arange(10).reshape(2, 5)
    tile_shape = (3, 3)
    tiles = break_into_tiles(sample, tile_shape)

    assert tiles.shape == (1, 2)
    np.testing.assert_array_equal(tiles[0, 0], np.array([[0, 1, 2], [5, 6, 7]]))
    np.testing.assert_array_equal(tiles[0, 1], np.array([[3, 4], [8, 9]]))

    # TODO: coalesce tiles into sample and compare


def test_serialize_tiles():
    sample = np.arange(10).reshape(2, 5)
    tile_shape = (3, 3)
    tiles = break_into_tiles(sample, tile_shape)
    serialized_tiles = serialize_tiles(tiles, lambda x: x.tobytes())

    assert serialized_tiles.shape == tiles.shape

    # TODO: deserialize and coalesce tiles into sample and compare


def test_serialize_tiles_gzip():
    sample = np.arange(10).reshape(2, 5)
    tile_shape = (3, 3)
    tiles = break_into_tiles(sample, tile_shape)

    gzip_compress = lambda x: gzip.compress(x.tobytes())
    serialized_tiles = serialize_tiles(tiles, gzip_compress)

    assert serialized_tiles.shape == tiles.shape