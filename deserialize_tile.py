from typing import Callable, Tuple, Union
import numpy as np


from tile_util import tile_bounds, validate_not_serialized, view


def coalesce_tiles(tiles: np.ndarray, tile_shape: Tuple[int, ...], sample_shape: Tuple[int, ...], dtype: Union[str, np.dtype]) -> np.ndarray:
    """Coalesce tiles into a single array of shape `sample_shape`.

    Args:
        tiles (np.ndarray): numpy object array of tiles.
        tile_shape (Tuple[int, ...]): Tile shape. Corner tiles may be smaller than this.
        sample_shape (Tuple[int, ...]): Shape of the output array. The sum of all actual tile shapes are expected to be equal to this.
        dtype (Union[str, np.dtype]): Dtype of the output array. Should match dtype of tiles.

    Raises:
        TypeError: If `tiles` is not deserialized.

    Returns:
        np.ndarray: Sample array from tiles.
    """

    sample = np.empty(sample_shape, dtype=dtype)
    if tiles.size <= 0:
        return sample

    validate_not_serialized(tiles, "coalescing tiles")

    for tile_coords, tile in np.ndenumerate(tiles):
        low, high = tile_bounds(np.asarray(tile_coords), tile_shape)
        sample_view = view(sample, low, high)

        sample_view[:] = tile

    return sample


def deserialize_tiles(serialized_tiles: np.ndarray, tile_shapes: np.ndarray, frombytes_func: Callable[[bytes], np.ndarray]) -> np.ndarray:
    # TODO: docstring, explain why tile_shapes is different from tile_shape (maybe rename)

    deserialized_tiles = np.empty(serialized_tiles.shape, dtype=object)

    for tile_coord, serialized_tile in np.ndenumerate(serialized_tiles):
        shape = tile_shapes[tile_coord]
        tile = frombytes_func(serialized_tile).reshape(shape)
        deserialized_tiles[tile_coord] = tile

    return deserialized_tiles