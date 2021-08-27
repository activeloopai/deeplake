from hub.core.compression import COMPRESSION_FACTORS, get_compression_factor
from hub.core.meta.tensor_meta import TensorMeta
from hub.constants import MB
from typing import Tuple
import numpy as np


# number of bytes tiles can be off by in comparison to max_chunk_size
COST_THRESHOLD = 1 * MB  # TODO: make smaller with better optimizers


def _cost(tile_shape: Tuple[int, ...], dtype: np.dtype, tile_target_bytes: int, compression_factor: float) -> int:
    # TODO: docstring

    actual_bytes = np.prod(tile_shape) * dtype.itemsize // compression_factor
    return abs(tile_target_bytes - actual_bytes)


def _downscale(tile_shape: Tuple[int, ...], sample_shape: Tuple[int, ...]):
    # TODO: docstring

    return tuple(map(min, zip(tile_shape, sample_shape)))


def _propose_tile_shape(
    sample_shape: Tuple[int, ...], dtype: np.dtype, max_chunk_size: int, compression_factor: float
):
    # TODO: docstring (use args)

    dtype_num_bytes = dtype.itemsize
    ndims = len(sample_shape)

    # TODO: factor in sample/chunk compression (ALSO DO THIS BEFORE MERGING)
    tile_target_entries = max_chunk_size // dtype_num_bytes
    tile_target_entries *= compression_factor
    tile_target_length = int(tile_target_entries ** (1.0 / ndims))

    # Tile length must be at least 16, so we don't round too close to zero
    tile_shape = max((16,) * ndims, (tile_target_length,) * ndims)

    return tile_shape


def _optimize_tile_shape(
    sample_shape: Tuple[int, ...], dtype: np.dtype, max_chunk_size: int, compression_factor: float
):
    # TODO: docstring

    tile_shape = _propose_tile_shape(sample_shape, dtype, max_chunk_size, compression_factor)

    downscaled_tile_shape = _downscale(tile_shape, sample_shape)
    if tile_shape == downscaled_tile_shape:
        return tile_shape

    # TODO
    raise NotImplementedError("Iterative optimization of tile shapes not yet implemented.")

    frozen_dims_mask = np.array(tile_shape) == np.array(downscaled_tile_shape)

    # iterate until we find a tile shape close to max chunk size
    proposed_tile_shape = downscaled_tile_shape
    while _cost(proposed_tile_shape, dtype, compression_factor) > COST_THRESHOLD:


        break

    return tile_shape

    # TODO: rescale up (must do before merging)
    raise NotImplementedError
    return tile_shape


def _validate_tile_shape(
    tile_shape: Tuple[int, ...], dtype: np.dtype, max_chunk_size: int, compression_factor: float
):
    # TODO: docstring

    cost = _cost(tile_shape, dtype, max_chunk_size, compression_factor)
    if cost > COST_THRESHOLD:
        raise Exception(
            f"Cost too large ({cost}) for tile shape {tile_shape} and max chunk size {max_chunk_size}"
        )  # TODO: exceptions.py


def optimize_tile_shape(sample_shape: Tuple[int, ...], tensor_meta: TensorMeta):
    # TODO: docstring
    
    dtype = np.dtype(tensor_meta.dtype)
    max_chunk_size = tensor_meta.max_chunk_size
    compression_factor = get_compression_factor(tensor_meta)

    tile_shape = _optimize_tile_shape(sample_shape, dtype, max_chunk_size, compression_factor)
    _validate_tile_shape(tile_shape, dtype, max_chunk_size, compression_factor)
    return tile_shape
