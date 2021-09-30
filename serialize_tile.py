from typing import Tuple, Union
from itertools import product
import numpy as np


def ceildiv(a: Union[np.ndarray, int, float], b: Union[np.ndarray, int, float]) -> Union[np.ndarray, int, float]:
    """General ceiling division."""

    return -(-a // b)



def view(sample: np.ndarray, low: Tuple[int, ...], high: Tuple[int, ...], step=1) -> np.ndarray:

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
    # TODO: docstring

    tiles_per_dim = ceildiv(np.array(sample.shape), tile_shape) # np.array(sample.shape) // tile_shape
    tiles = np.empty(tiles_per_dim, dtype=object)

    for tile_coord, _ in np.ndenumerate(tiles):
        tile_coord_arr = np.asarray(tile_coord)

        low = tuple(tile_coord_arr * tile_shape)
        high = tuple((tile_coord_arr + 1) * tile_shape)

        tile = view(sample, low, high)

        tiles[tile_coord] = tile

    # return numpy array of numpy objects (tiles) in tile order
    return tiles


def serialize_tiles(tiles: np.ndarray) -> np.ndarray:
    # TODO: docstring

    # accept numpy array of numpy objects (tiles) in tile order

    # return byte object numpy array, compressed if applicable

    pass


def test():
    sample = np.arange(10).reshape(2, 5)
    tile_shape = (3, 3)
    tiles = break_into_tiles(sample, tile_shape)

    assert tiles.shape == (1, 2)
    np.testing.assert_array_equal(tiles[0, 0], np.array([[0, 1, 2], [5, 6, 7]]))
    np.testing.assert_array_equal(tiles[0, 1], np.array([[3, 4], [8, 9]]))


if __name__ == "__main__":
    test()