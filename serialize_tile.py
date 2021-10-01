from typing import Callable, Tuple
import numpy as np

from tile_util import ceildiv, tile_bounds, validate_not_serialized, view


def break_into_tiles(sample: np.ndarray, tile_shape: Tuple[int, ...]) -> np.ndarray:
    """Get a new tile-ordered numpy object array that is the shape of the tile grid.
    Each element of the returned numpy object array is also a numpy array that is a zero-copy view of the actual tile
    at the tile coordinate.

    Example:
        >>> tiles = break_into_tiles(np.arange(10).reshape(2, 5), (3, 3))
        >>> tiles
        [[array([[0, 1, 2],
                 [5, 6, 7]])
          array([[3, 4],
                 [8, 9]])]]
        >>> tiles.shape
        (1, 2)
        >>> tiles[0, 0]
        array([[0, 1, 2],
               [5, 6, 7]])
        >>> tiles[0, 1]
        array([[3, 4],
               [8, 9]])

    Args:
        sample: The sample array to break into tiles.
        tile_shape: The shape that each tile will have. Determines the number of tiles in each dimension.
            Corner tiles (tiles at the end of one or more dimensions) may be smaller than the tile shape.

    Returns:
        numpy object array of tiles. Each element of the array is a numpy array that is a zero-copy view (source = sample array)
            of the actual tile at the tile coordinate.
    """

    tiles_per_dim = ceildiv(np.array(sample.shape), tile_shape)
    tiles = np.empty(tiles_per_dim, dtype=object)

    for tile_coord, _ in np.ndenumerate(tiles):
        tile_coord_arr = np.asarray(tile_coord)

        low, high = tile_bounds(tile_coord_arr, tile_shape)
        tile = view(sample, low, high)

        tiles[tile_coord] = tile

    return tiles


def serialize_tiles(tiles: np.ndarray, tobytes_func: Callable[[np.ndarray], bytes]) -> np.ndarray:
    """Get a new tile-ordered numpy object array that is the same shape of the tile-grid.
    Each element of the returned numpy object array is a bytes object representing the serialized tile.

    Args:
        tiles (np.ndarray): The tile-ordered numpy object array to serialize.
        tobytes_func (Callable[[np.ndarray], bytes]): A function that takes a numpy array and returns a bytes object.
            This function is used to serialize each tile, may be used to compress the tile.

    Returns:
        np.ndarray: numpy object array of serialized tiles. Each element of the array is a bytes object.
    """

    # TODO: maybe use memoryview and update docstring

    validate_not_serialized(tiles, "serialize_tiles")

    serialized_tiles = np.empty(tiles.shape, dtype=object)
    for tile_coord, tile in np.ndenumerate(tiles):
        serialized_tiles[tile_coord] = tobytes_func(tile)
    return serialized_tiles
