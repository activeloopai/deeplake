import numpy as np
from typing import List, Tuple, Union
from hub.core.chunk.base_chunk import BaseChunk
from hub.core.meta.encode.tile import TileEncoder
from hub.core.tiling.util import tile_bounds, validate_not_serialized, view


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


def combine_chunks(
    chunks: List[BaseChunk], sample_index: int, tile_encoder: TileEncoder
) -> np.ndarray:
    dtype = chunks[0].dtype
    shape = tile_encoder.get_sample_shape(sample_index)
    tile_shape = tile_encoder.get_tile_shape(sample_index)
    layout_shape = tile_encoder.get_tile_layout_shape(sample_index)

    # index is always 0 within a chunk for tiled samples
    tiled_arrays = [chunk.read_sample(0) for chunk in chunks]
    return np_list_to_sample(tiled_arrays, shape, tile_shape, layout_shape, dtype)


def np_list_to_sample(
    tiled_arrays: List[np.ndarray], shape, tile_shape, layout_shape, dtype
) -> np.ndarray:
    num_tiles = len(tiled_arrays)
    tiles = np.empty((num_tiles,), dtype=object)
    tiles[:] = tiled_arrays[:]
    tiles = np.reshape(tiles, layout_shape)
    return coalesce_tiles(tiles, tile_shape, shape, dtype)
