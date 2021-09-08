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
    factor = compression_factor
    return int(num_bytes // factor)


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


def modified_space_subslice(array: np.ndarray, subslice_index: Index, min_bound: Tuple[int, ...], max_bound: Tuple[int, ...]):
    # TODO: docstring/rename method

    # return a view of the array where the coordinate system may be broader or narrower than the actual array's shape

    # TODO: make arguments optional?

    subslice_min = subslice_index.low_bound  # TODO rename to min/max
    subslice_max = subslice_index.high_bound

    # sanity checks
    if len(array.shape) != len(subslice_min) or len(subslice_min) != len(max_bound):
        raise Exception()  # TODO: exceptions 

    bias_shape = np.array(array.shape) + subslice_min
    
    # get sample-coordinates for the overlap with tile-coordinates
    low_corner = np.maximum(subslice_min, min_bound)
    high_corner = np.minimum(subslice_max, max_bound)
    delta = high_corner - low_corner

    # get the subslice of the sample for the overlap
    entries = []
    for dim in delta:
        entries.append(IndexEntry(slice(0, dim)))
    array_subslice_index = Index(entries)
    array_subslice = array_subslice_index.apply([array], include_first_value=True)[0]  # TODO: refac

    return array_subslice


def get_sample_subslice(sample: np.ndarray, tile_index: Tuple[int, ...], tile_shape_mask: np.ndarray, subslice_index: Index):
    """Get the subslice of a sample that is contained within the tile at `tile_index`.

    # TODO: examples

    Args:
        sample (np.ndarray): The full sample to get the subslice of.
        tile_index (Tuple[int, ...]): Index of the tile to get the subslice of.
        tile_shape_mask (np.ndarray): A mask of the shape of each tile.

    Returns:
        np.ndarray: The subslice of the sample.
    """

    # TODO: can remove this check (maybe move it higher up in the stack)
    if not np.all(tile_shape_mask):
        # sanity check
        raise NotImplementedError("Cannot handle dynamic tile shapes yet!")

    # get tile bounds
    tile_shape = tile_shape_mask[tile_index]
    tile_low_bound, tile_high_bound = get_tile_bounds(tile_index, tile_shape)

    return modified_space_subslice(sample, subslice_index, tile_low_bound, tile_high_bound)



def align_sample_and_tile(sample: np.ndarray, tile: np.ndarray, subslice_index: Index, tile_index: Tuple[int, ...]=None):
    # TODO: docstring

    # tile index should be `None` if the sample is not tiled
    if tile_index is None:
        tile_index = (0,) * len(tile.shape)

    low, high = get_tile_bounds(
        tile_index, tile.shape
    )

    tile_view = subslice_index.apply_restricted(tile, bias=low)
    incoming_sample_view = subslice_index.apply_restricted(sample, bias=low, upper_bound=high, normalize=True)

    return tile_view, incoming_sample_view
