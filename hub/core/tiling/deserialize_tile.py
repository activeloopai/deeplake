import numpy as np
from typing import Tuple, Union
from hub.core.tiling.util import tile_bounds, validate_not_serialized, view  # type: ignore


def coalesce_tiles(
    tiles: np.ndarray,
    tile_shape: Tuple[int, ...],
    sample_shape: Tuple[int, ...],
    dtype: Union[str, np.dtype],
) -> np.ndarray:
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
