from typing import Tuple, Union
from itertools import product
import numpy as np


def ceildiv(
    a: Union[np.ndarray, int, float], b: Union[np.ndarray, int, float]
) -> Union[np.ndarray, int, float]:
    """General ceiling division."""

    return -(-a // b)


def view(
    sample: np.ndarray, low: Tuple[int, ...], high: Tuple[int, ...], step=1
) -> np.ndarray:
    """Get a zero-copy view of the sample array on the n-dimensional interval low:high.

    For a 3D sample, the indexing would look like this:
        sample[low[0]:high[0], low[1]:high[1], low[2]:high[2]]
    """

    if len(sample.shape) != len(low) or len(sample.shape) != len(high):
        raise ValueError(
            "low, high, and sample must have the same number of dimensions"
        )

    slices = []
    for low_dim, high_dim in zip(low, high):
        if low_dim < 0 or high_dim < 0:
            raise ValueError("Low/high dim must be >= 0.")
        elif low_dim > high_dim:
            raise ValueError("Low dim must be <= high dim.")

        slices.append(slice(low_dim, high_dim, step))

    sample_view = sample[tuple(slices)]
    return sample_view


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

    tiles_per_dim = ceildiv(
        np.array(sample.shape), tile_shape
    )  # np.array(sample.shape) // tile_shape
    tiles = np.empty(tiles_per_dim, dtype=object)

    for tile_coord, _ in np.ndenumerate(tiles):
        tile_coord_arr = np.asarray(tile_coord)

        low = tuple(tile_coord_arr * tile_shape)
        high = tuple((tile_coord_arr + 1) * tile_shape)

        tile = view(sample, low, high)

        tiles[tile_coord] = tile

    return tiles


def serialize_tiles(tiles: np.ndarray) -> np.ndarray:
    """Get a new tile-ordered numpy object array that is the same shape of the tile-grid.
    Each element of the returned numpy object array is a bytes object representing the serialized tile.
    """

    # TODO: compression and update docstring
    # TODO: maybe use memoryview and update docstring

    serialized_tiles = np.empty(tiles.shape, dtype=object)

    for tile_coord, tile in np.ndenumerate(tiles):
        serialized_tiles[tile_coord] = tile.tobytes()

    return serialized_tiles


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
    serialized_tiles = serialize_tiles(tiles)

    assert serialized_tiles.shape == tiles.shape

    # TODO: deserialize and coalesce tiles into sample and compare


if __name__ == "__main__":
    test_break_into_tiles()
    test_serialize_tiles()
