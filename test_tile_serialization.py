import numpy as np
import gzip

from serialize_tile import *
from deserialize_tile import *


def get_arange_sample(shape: Tuple[int, ...]) -> np.ndarray:
    return np.arange(np.prod(shape)).reshape(*shape)


def test_break_into_tiles():
    sample = get_arange_sample((2, 5))
    tile_shape = (3, 3)

    tiles = break_into_tiles(sample, tile_shape)
    assert tiles.shape == (1, 2)
    np.testing.assert_array_equal(tiles[0, 0], np.array([[0, 1, 2], [5, 6, 7]]))
    np.testing.assert_array_equal(tiles[0, 1], np.array([[3, 4], [8, 9]]))

    coalesced_sample = coalesce_tiles(tiles, sample.shape)
    np.testing.assert_array_equal(sample, coalesced_sample)


def test_serialize_tiles():
    sample = get_arange_sample((2, 5))
    tile_shape = (3, 3)

    tiles = break_into_tiles(sample, tile_shape)
    serialized_tiles = serialize_tiles(tiles, lambda x: x.tobytes())
    assert serialized_tiles.shape == tiles.shape

    deserialized_tiles = deserialize_tiles(serialized_tiles, lambda x: np.frombuffer(x, dtype=sample.dtype))
    coalesced_sample = coalesce_tiles(deserialized_tiles, sample.shape)
    np.testing.assert_array_equal(sample, coalesced_sample)


def test_serialize_tiles_gzip():
    sample = get_arange_sample((2, 5))
    tile_shape = (3, 3)

    tiles = break_into_tiles(sample, tile_shape)
    gzip_compress = lambda x: gzip.compress(x.tobytes())
    serialized_tiles = serialize_tiles(tiles, gzip_compress)
    assert serialized_tiles.shape == tiles.shape

    gzip_decompress = lambda x: np.frombuffer(gzip.decompress(x), dtype=sample.dtype)
    deserialized_tiles = deserialize_tiles(serialized_tiles, gzip_decompress)
    coalesced_sample = coalesce_tiles(deserialized_tiles, sample.shape)
    np.testing.assert_array_equal(sample, coalesced_sample)