from typing import Tuple
import numpy as np
import gzip

from deeplake.core.tiling.serialize import break_into_tiles, serialize_tiles
from deeplake.core.tiling.deserialize import coalesce_tiles, np_list_to_sample


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
    shapes = [t.shape for _, t in np.ndenumerate(tiles)]
    serialized_tiles = serialize_tiles(tiles, lambda x: memoryview(x.tobytes()))
    assert serialized_tiles.shape == tiles.shape

    flattened_tiles = serialized_tiles.reshape((serialized_tiles.size,))
    tiled_arrays = [
        np.frombuffer(x, dtype=sample.dtype).reshape(sh)
        for x, sh in zip(flattened_tiles, shapes)
    ]
    coalesced_sample = np_list_to_sample(
        tiled_arrays, sample.shape, tile_shape, tiles.shape, sample.dtype
    )
    np.testing.assert_array_equal(sample, coalesced_sample)


def test_serialize_tiles_gzip():
    sample = _get_arange_sample((2, 5))
    tile_shape = (3, 3)

    tiles = break_into_tiles(sample, tile_shape)
    shapes = [t.shape for _, t in np.ndenumerate(tiles)]
    gzip_compress = lambda x: memoryview(gzip.compress(x.tobytes()))
    serialized_tiles = serialize_tiles(tiles, gzip_compress)
    assert serialized_tiles.shape == tiles.shape

    flattened_tiles = serialized_tiles.reshape((serialized_tiles.size,))
    gzip_decompress = lambda x: np.frombuffer(gzip.decompress(x), dtype=sample.dtype)
    tiled_arrays = [
        gzip_decompress(x).reshape(sh) for x, sh in zip(flattened_tiles, shapes)
    ]

    coalesced_sample = np_list_to_sample(
        tiled_arrays, sample.shape, tile_shape, tiles.shape, sample.dtype
    )
    np.testing.assert_array_equal(sample, coalesced_sample)
