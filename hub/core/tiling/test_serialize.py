import gzip
from numpy import (
    ndarray,
    ndenumerate,
    frombuffer,
    array as np_array,
    prod as np_prod,
    arange as np_arange,
    testing as np_testing,
)
from typing import Tuple

from hub.core.tiling.serialize import break_into_tiles, serialize_tiles
from hub.core.tiling.deserialize import coalesce_tiles, np_list_to_sample


def _get_arange_sample(shape: Tuple[int, ...]) -> ndarray:
    return np_arange(np_prod(shape)).reshape(*shape)


def test_break_into_tiles():
    sample = _get_arange_sample((2, 5))
    tile_shape = (3, 3)

    tiles = break_into_tiles(sample, tile_shape)
    assert tiles.shape == (1, 2)
    np_testing.assert_array_equal(tiles[0, 0], np_array([[0, 1, 2], [5, 6, 7]]))
    np_testing.assert_array_equal(tiles[0, 1], np_array([[3, 4], [8, 9]]))

    coalesced_sample = coalesce_tiles(tiles, tile_shape, sample.shape, sample.dtype)
    np_testing.assert_array_equal(sample, coalesced_sample)


def test_serialize_tiles():
    sample = _get_arange_sample((2, 5))
    tile_shape = (3, 3)

    tiles = break_into_tiles(sample, tile_shape)
    shapes = [t.shape for _, t in ndenumerate(tiles)]
    serialized_tiles = serialize_tiles(tiles, lambda x: memoryview(x.tobytes()))
    assert serialized_tiles.shape == tiles.shape

    flattened_tiles = serialized_tiles.reshape((serialized_tiles.size,))
    tiled_arrays = [
        frombuffer(x, dtype=sample.dtype).reshape(sh)
        for x, sh in zip(flattened_tiles, shapes)
    ]
    coalesced_sample = np_list_to_sample(
        tiled_arrays, sample.shape, tile_shape, tiles.shape, sample.dtype
    )
    np_testing.assert_array_equal(sample, coalesced_sample)


def test_serialize_tiles_gzip():
    sample = _get_arange_sample((2, 5))
    tile_shape = (3, 3)

    tiles = break_into_tiles(sample, tile_shape)
    shapes = [t.shape for _, t in ndenumerate(tiles)]
    gzip_compress = lambda x: memoryview(gzip.compress(x.tobytes()))
    serialized_tiles = serialize_tiles(tiles, gzip_compress)
    assert serialized_tiles.shape == tiles.shape

    flattened_tiles = serialized_tiles.reshape((serialized_tiles.size,))
    gzip_decompress = lambda x: frombuffer(gzip.decompress(x), dtype=sample.dtype)
    tiled_arrays = [
        gzip_decompress(x).reshape(sh) for x, sh in zip(flattened_tiles, shapes)
    ]

    coalesced_sample = np_list_to_sample(
        tiled_arrays, sample.shape, tile_shape, tiles.shape, sample.dtype
    )
    np_testing.assert_array_equal(sample, coalesced_sample)
