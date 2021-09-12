from hub.core.index.index import Index, IndexEntry
from math import ceil
from typing import Tuple
import numpy as np


def num_bytes_without_compression(shape: Tuple[int, ...], dtype: np.dtype) -> int:
    return dtype.itemsize * np.prod(shape)


def approximate_num_bytes(shape, dtype, compression_factor: float) -> int:
    """Calculate the number of bytes required to store raw data with the given shape. If no compression is used, this will be an exact
    number of bytes. If compressed, it will be approximated assuming the data is natural."""

    num_bytes = num_bytes_without_compression(shape, np.dtype(dtype))
    return int(num_bytes // compression_factor)


def ceildiv(a: int, b: int) -> int:
    """Computes the ceiling of the division of two ints.
    Returns an int.
    """

    return ceil(float(a) / float(b))


def num_tiles_for_sample(
    tile_shape: Tuple[int, ...], sample_shape: Tuple[int, ...]
) -> int:
    """Calculates the number of tiles required to store a sample of `sample_shape` using tiles
    of shape `tile_shape`."""

    num_tiles = 1
    for tile_dim, sample_dim in zip(tile_shape, sample_shape):
        num_tiles *= ceildiv(sample_dim, tile_dim)
    return num_tiles


def get_tile_bounds(
    tile_index: Tuple[int, ...], tile_shape: Tuple[int, ...]
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    # TODO: docstring

    if tile_index is None:
        tile_index = (0,) * len(tile_shape)

    # TODO: note: this ONLY works when tile_shapes are uniform for a sample!
    low, high = [], []

    for index_dim, shape_dim in zip(tile_index, tile_shape):
        low.append(index_dim * shape_dim)
        high.append((index_dim + 1) * shape_dim)

    return tuple(low), tuple(high)


def get_tile_mask(
    ordered_tile_ids: np.ndarray, tile_shape_mask: np.ndarray, subslice_index: Index
):
    # loop through each tile ID, check if it exists within the subslice_index.

    mask = np.zeros(ordered_tile_ids.shape, dtype=bool)

    if ordered_tile_ids.size == 1:
        mask[:] = True
        return mask

    for tile_index, _ in np.ndenumerate(ordered_tile_ids):
        tile_shape = tile_shape_mask[tile_index]
        low, high = get_tile_bounds(tile_index, tile_shape)

        if subslice_index.intersects(low, high):
            mask[tile_index] = True

    return mask


def align_sample_and_tile(
    sample: np.ndarray,
    tile: np.ndarray,
    subslice_index: Index,
    tile_index: Tuple[int, ...] = None,
):
    # TODO: docstring

    # tile index should be `None` if the sample is not tiled
    if tile_index is None:
        tile_index = (0,) * len(tile.shape)

    low, high = get_tile_bounds(tile_index, tile.shape)

    tile_view = subslice_index.apply_restricted(tile, bias=low)
    incoming_sample_view = subslice_index.apply_restricted(
        sample, bias=low, upper_bound=high, normalize=True
    )

    return tile_view, incoming_sample_view


def get_input_tile_view(
    tile: np.ndarray,
    subslice_index: Index,
    tile_index: Tuple[int, ...],
    tile_shape: Tuple[int, ...],
) -> np.ndarray:
    # TODO: docstring (mention why tile_shape is not the same as tile.shape sometimes)

    low, high = get_tile_bounds(tile_index, tile_shape)

    return subslice_index.apply_restricted(tile, bias=low)


def get_output_tile_view(
    tile: np.ndarray,
    subslice_index: Index,
    tile_index: Tuple[int, ...],
    tile_shape: Tuple[int, ...],
) -> np.ndarray:

    low, high = get_tile_bounds(tile_index, tile_shape)
    bias = np.asarray(low)
    return subslice_index.apply_restricted(tile, bias=bias)


def get_input_sample_view(
    sample: np.ndarray,
    subslice_index: Index,
    tile_index: Tuple[int, ...],
    tile_shape: Tuple[int, ...],
) -> np.ndarray:

    low, high = get_tile_bounds(tile_index, tile_shape)
    return subslice_index.apply_restricted(
        sample, bias=low, upper_bound=high, normalize=True
    )


def get_output_sample_view(
    sample: np.ndarray,
    subslice_index: Index,
    tile_index: Tuple[int, ...],
    tile_shape: Tuple[int, ...],
) -> np.ndarray:
    """When reading a subslice from a tiled sample, this method restricts the view on `sample` 
    (so it can be populated with data from a tile) where `sample` is the array that will be returned 
    to the user. Tile data has it's view restricted and then all the tiles data are gathered into `sample`."""

    low, high = get_tile_bounds(tile_index, tile_shape)
    bias = np.asarray(low)
    return subslice_index.apply_restricted(sample, bias=bias, upper_bound=high, normalize=True)

    subslice_index.add_trivials(len(sample.shape))
    
    low, high = get_tile_bounds(tile_index, tile_shape)

    slices = []
    for low_dim, high_dim in zip(low, high):
        slices.append(slice(low_dim, high_dim))
    slices = tuple(slices)

    print(slices)

    return sample[slices]


def get_tile_view_on_sample(
    sample: np.ndarray, tile_shape: Tuple[int, ...], tile_index: Tuple[int, ...]
) -> np.ndarray:
    # TODO: docstring (used for breaking sample into tiles)

    # TODO: maybe this method is duplicate logic?

    low, high = get_tile_bounds(tile_index, tile_shape)

    slices = []
    for low_dim, high_dim in zip(low, high):
        slices.append(slice(low_dim, high_dim))
    slices = tuple(slices)

    return sample[slices]
