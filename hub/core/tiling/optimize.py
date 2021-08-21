from hub.constants import MB
from typing import Tuple
import numpy as np


# number of bytes tiles can be off by in comparison to max_chunk_size
COST_THRESHOLD = 1 * MB  # TODO: make smaller with better optimizers


def _cost(tile_shape: Tuple[int], dtype: np.dtype, tile_target_bytes: int) -> int:
    # TODO: docstring

    actual_bytes = np.prod(tile_shape) * dtype.itemsize
    return abs(tile_target_bytes - actual_bytes)


def _downscale(tile_shape: Tuple[int], sample_shape: Tuple[int]):
    # TODO: docstring
    
    return tuple(map(min, zip(tile_shape, sample_shape)))


def _propose_tile_shape(sample_shape: Tuple[int], dtype: np.dtype, max_chunk_size: int):
    dtype_num_bytes = dtype.itemsize
    ndims = len(sample_shape)

    # TODO: factor in sample/chunk compression (ALSO DO THIS BEFORE MERGING)
    tile_target_entries = max_chunk_size // dtype_num_bytes
    tile_target_length = int(tile_target_entries ** (1.0 / ndims))

    # Tile length must be at least 16, so we don't round too close to zero
    tile_shape = max((16,) * ndims, (tile_target_length,) * ndims)

    return tile_shape


def _optimize_tile_shape(sample_shape: Tuple[int], dtype: np.dtype, max_chunk_size: int):
    # TODO: docstring

    tile_shape = _propose_tile_shape(sample_shape, dtype, max_chunk_size)

    downscaled_tile_shape = _downscale(tile_shape, sample_shape)
    if tile_shape == downscaled_tile_shape:
        return tile_shape

    # TODO: rescale up (must do before merging)
    raise NotImplementedError
    return tile_shape


def _validate_tile_shape(tile_shape: Tuple[int], dtype: np.dtype, max_chunk_size: int):
    # TODO: docstring

    cost = _cost(tile_shape, dtype, max_chunk_size)
    if cost > COST_THRESHOLD:
        raise Exception(f"Cost too large ({cost}) for tile shape {tile_shape} and max chunk size {max_chunk_size}")  # TODO: exceptions.py


def get_tile_shape(sample_shape: Tuple[int], dtype: np.dtype, max_chunk_size: int):
    # TODO: docstring

    tile_shape = _optimize_tile_shape(sample_shape, dtype, max_chunk_size)
    _validate_tile_shape(tile_shape, dtype, max_chunk_size)
    return tile_shape
