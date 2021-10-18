import numpy as np
from typing import Union, Tuple


def ceildiv(
    a: Union[np.ndarray, int, float], b: Union[np.ndarray, int, float]
) -> Union[np.ndarray, int, float]:
    """General ceiling division."""

    return np.ceil(a / b).astype(int)


def validate_not_serialized(tiles: np.ndarray, operation: str):
    if type(tiles.flatten()[0]) == memoryview:
        raise TypeError(f"Before {operation}, you should deserialize the tiles.")


def validate_is_serialized(tiles: np.ndarray, operation: str):
    if type(tiles.flatten()[0]) == np.ndarray:
        raise TypeError(f"Before {operation}, you should serialize the tiles.")


def tile_bounds(
    tile_coords: np.ndarray, tile_shape: Tuple[int, ...]
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Returns the low and high bounds of the tile at `tile_coords` assuming all tiles are of shape `tile_shape`."""

    if len(tile_coords) != len(tile_shape):
        raise ValueError(
            "tile_coords and tile_shape must have the same number of dimensions."
        )

    low = tuple(tile_coords * tile_shape)
    high = tuple((tile_coords + 1) * tile_shape)
    return low, high


def get_tile_shapes(tiles: np.ndarray) -> np.ndarray:
    """Returns a numpy object array with the same shape as `tiles` where each value is a tuple representing each tiles respective shape.
    Different from the overall `tile_shape`, these tile shapes may be smaller (especially for the corner tiles).
    """

    validate_not_serialized(tiles, "getting tile shapes")

    tile_shapes = np.empty(tiles.shape, dtype=object)
    for tile_coord, tile in np.ndenumerate(tiles):
        tile_shapes[tile_coord] = tile.shape
    return tile_shapes


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
        if low_dim > high_dim:
            raise ValueError("Low dim must be <= high dim.")

        slices.append(slice(low_dim, high_dim, step))

    sample_view = sample[tuple(slices)]
    return sample_view
