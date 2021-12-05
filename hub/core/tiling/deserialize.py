import numpy as np
from typing import List, Tuple, Union
from hub.core.chunk.base_chunk import BaseChunk
from hub.core.meta.encode.tile import TileEncoder
from hub.core.tiling.util import tile_bounds, validate_not_serialized, view


def coalesce_tiles(
    tiles: np.ndarray,
    tile_shape: Tuple[int, ...],
    dtype: Union[str, np.dtype],
) -> np.ndarray:
    """Coalesce tiles into a single array
    Args:
        tiles (np.ndarray): numpy object array of tiles.
        tile_shape (Tuple[int, ...]): Tile shape. Corner tiles may be smaller than this.
        dtype (Union[str, np.dtype]): Dtype of the output array. Should match dtype of tiles.
    Raises:
        TypeError: If `tiles` is not deserialized.
    Returns:
        np.ndarray: Sample array from tiles.
    """

    sample = np.empty(np.multiply(tiles.shape, tile_shape), dtype=dtype)
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
    tile_shape = tile_encoder.get_tile_shape(sample_index)
    layout_shape = tile_encoder.get_tile_layout_shape(sample_index)

    # index is always 0 within a chunk for tiled samples
    tiled_arrays = [chunk.read_sample(0) for chunk in chunks]
    return np_list_to_sample(tiled_arrays, tile_shape, layout_shape, dtype)


def np_list_to_sample(
    tiled_arrays: List[np.ndarray], tile_shape, layout_shape, dtype
) -> np.ndarray:
    num_tiles = len(tiled_arrays)
    tiles = np.empty((num_tiles,), dtype=object)
    tiles[:] = tiled_arrays[:]
    tiles = np.reshape(tiles, layout_shape)
    return coalesce_tiles(tiles, tile_shape, dtype)


def translate_slices(
    slices: List[Union[slice, int, List[int]]],
    sample_shape: Tuple[int, ...],
    tile_shape: Tuple[int, ...],
) -> Tuple[Tuple, Tuple]:
    """Translates slices from sample space to tile space
    Args:
        sample_shape (Tuple[int, ...]): Sample shape.
        tile_shape (Tuple[int, ...]): Tile shape.
    Raises:
        NotImplementedError: For stepping slices
    """
    tiles_index = []
    sample_index = []
    for i, s in enumerate(slices):
        if isinstance(s, int):
            ts = (s + sample_shape[i] if s < 0 else s) // tile_shape[i]
            tiles_index.append(slice(ts, ts + 1))
            sample_index.append(0)
        elif isinstance(s, list):
            s = [x + sample_shape[i] if x < 0 else x for x in s]
            mn, mx = min(s), max(s)
            tiles_index.append(slice(mn // tile_shape[i], mx // tile_shape[i] + 1))
            sample_index.append([x - mn for x in s])
        elif isinstance(s, slice):
            start, stop, step = s.start, s.stop, s.step
            if start is None:
                start = 0
            elif start < 0:
                start += sample_shape[i]
            if stop is None:
                stop = sample_shape[i]
            elif stop < 0:
                stop += sample_shape[i]
            else:
                stop = min(stop, sample_shape[i])
            if step not in (1, None):
                raise NotImplementedError(
                    "Stepped indexing for tiled samples is not supported yet."
                )
            ts = slice(start // tile_shape[i], (stop - 1) // tile_shape[i] + 1)
            tiles_index.append(ts)
            offset = ts.start * tile_shape[i]
            sample_index.append(slice(start - offset, stop - offset))
    return tuple(tiles_index), tuple(sample_index)
