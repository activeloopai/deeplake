from numpy import (
    ndarray,
    ndenumerate,
    empty as np_empty,
    ceil as np_ceil,
    divide as np_divide,
    multiply as np_multiply,
    vectorize as np_vectorize,
)
from typing import Callable, Tuple


def break_into_tiles(sample: ndarray, tile_shape: Tuple[int, ...]) -> ndarray:
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

    tiles_per_dim = np_ceil(np_divide(sample.shape, tile_shape)).astype(int)
    tiles = np_empty(tiles_per_dim, dtype=object)
    for tile_coords, _ in ndenumerate(tiles):
        low = np_multiply(tile_coords, tile_shape)
        high = low + tile_shape
        idx = tuple(slice(l, h) for l, h in zip(low, high))
        tiles[tile_coords] = sample[idx]
    return tiles


def serialize_tiles(
    tiles: ndarray, serialize_func: Callable[[ndarray], memoryview]
) -> ndarray:
    """Get a new tile-ordered numpy object array that is the same shape of the tile-grid.
    Each element of the returned numpy object array is a memoryview object representing the serialized tile.

    Args:
        tiles (np.ndarray): The tile-ordered numpy object array to serialize.
        serialize_func (Callable[[np.ndarray], memoryview]): A function that takes a numpy array and returns a memoryview object.
            This function is used to serialize each tile, may be used to compress the tile.

    Returns:
        np.ndarray: numpy object array of serialized tiles. Each element of the array is a memoryview object.
    """
    return np_vectorize(serialize_func, otypes=[object])(tiles)
