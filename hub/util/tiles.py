from hub.core.index.index import Index
from math import ceil
from hub.core.meta.tensor_meta import TensorMeta
from typing import Tuple
import numpy as np


def _num_bytes_without_compression(shape: Tuple[int, ...], dtype: np.dtype) -> int:
    return dtype.itemsize * np.prod(shape)


def _approximate_num_bytes_after_compressing(num_bytes: int, compressor: str) -> int:
    # TODO: come up with better approximations for compression amount, also add support for all of our compressors
    if compressor == "png":
        return num_bytes // 2
    elif compressor == "mp4":
        return num_bytes // 100
    else:
        raise NotImplementedError


def approximate_num_bytes(shape, tensor_meta: TensorMeta) -> int:
    """Calculate the number of bytes required to store raw data with the given shape. If no compression is used, this will be an exact
    number of bytes. If compressed, it will be approximated assuming the data is natural."""

    num_bytes = _num_bytes_without_compression(shape, np.dtype(tensor_meta.dtype))

    # check sample compression first. we don't support compressing both sample + chunk-wise at the same time, but in case we
    # do support this in the future, try both.
    if tensor_meta.sample_compression is not None:
        num_bytes = _approximate_num_bytes_after_compressing(
            num_bytes, tensor_meta.sample_compression
        )

    # TODO: UNCOMMENT AFTER CHUNK-WISE COMPRESSION IS MERGED!
    # if tensor_meta.chunk_compression is not None:
    #     num_bytes = _approximate_num_bytes_after_compressing(
    #         num_bytes, tensor_meta.chunk_compression
    #     )

    return num_bytes


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


def get_tile_bounds(tile_index: Tuple[int], tile_shape: Tuple[int]) -> Tuple[Tuple[int], Tuple[int]]:
    low, high = [], []

    for index_dim, shape_dim in zip(tile_index, tile_shape):
        low.append(index_dim * shape_dim)
        high.append((index_dim + 1) * shape_dim)

    return tuple(low), tuple(high)



def get_tile_mask(ordered_tile_ids: np.ndarray, tile_shape_mask: np.ndarray, subslice_index: Index):
    # loop through each tile ID, check if it exists within the subslice_index.

    mask = np.zeros(ordered_tile_ids.shape, dtype=bool)

    for tile_index, _ in np.ndenumerate(ordered_tile_ids):
        tile_shape = tile_shape_mask[tile_index]
        low, high = get_tile_bounds(tile_index, tile_shape)

        if subslice_index.intersects(low, high):
            mask[tile_index] = True
            # chunk = self.get_chunk_from_id(tile_id)
            # tile_sample = self.read_sample_from_chunk(global_sample_index, chunk)

    return mask