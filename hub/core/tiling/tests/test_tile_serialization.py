from typing import Tuple
import pytest
import numpy as np
import gzip

from hub.core.tiling.serialize_tiles import break_into_tiles, serialize_tiles  # type: ignore
from hub.core.tiling.deserialize_tiles import coalesce_tiles, deserialize_tiles  # type: ignore
from hub.util.tiling import get_tile_shapes


def _get_arange_sample(shape: Tuple[int, ...]) -> np.ndarray:
    return np.arange(np.prod(shape)).reshape(*shape)


def test_break_into_tiles():
    sample = _get_arange_sample((2, 5))
    tile_shape = (3, 3)

    tiles = break_into_tiles(sample, tile_shape)
    assert tiles.shape == (1, 2)
    np.testing.assert_array_equal(tiles[0, 0], np.array([[0, 1, 2], [5, 6, 7]]))
    np.testing.assert_array_equal(tiles[0, 1], np.array([[3, 4], [8, 9]]))

    coalesced_sample = coalesce_tiles(tiles, tile_shape, sample.shape, sample.dtype)
    np.testing.assert_array_equal(sample, coalesced_sample)


def test_serialize_tiles():
    sample = _get_arange_sample((2, 5))
    tile_shape = (3, 3)

    tiles = break_into_tiles(sample, tile_shape)
    tile_shapes = get_tile_shapes(tiles)

    serialized_tiles = serialize_tiles(tiles, lambda x: memoryview(x.tobytes()))
    assert serialized_tiles.shape == tiles.shape

    with pytest.raises(TypeError):
        coalesce_tiles(serialized_tiles, tile_shape, sample.shape, sample.dtype)

    deserialized_tiles = deserialize_tiles(
        serialized_tiles, tile_shapes, lambda x: np.frombuffer(x, dtype=sample.dtype)
    )
    coalesced_sample = coalesce_tiles(
        deserialized_tiles, tile_shape, sample.shape, sample.dtype
    )
    np.testing.assert_array_equal(sample, coalesced_sample)


def test_serialize_tiles_gzip():
    sample = _get_arange_sample((2, 5))
    tile_shape = (3, 3)

    tiles = break_into_tiles(sample, tile_shape)
    tile_shapes = get_tile_shapes(tiles)

    gzip_compress = lambda x: memoryview(gzip.compress(x.tobytes()))
    serialized_tiles = serialize_tiles(tiles, gzip_compress)
    assert serialized_tiles.shape == tiles.shape

    gzip_decompress = lambda x: np.frombuffer(gzip.decompress(x), dtype=sample.dtype)
    deserialized_tiles = deserialize_tiles(
        serialized_tiles, tile_shapes, gzip_decompress
    )
    coalesced_sample = coalesce_tiles(
        deserialized_tiles, tile_shape, sample.shape, sample.dtype
    )
    np.testing.assert_array_equal(sample, coalesced_sample)
