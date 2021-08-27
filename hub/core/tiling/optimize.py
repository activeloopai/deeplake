from hub.core.compression import COMPRESSION_FACTORS, get_compression_factor
from hub.core.meta.tensor_meta import TensorMeta
from hub.constants import KB
from typing import Tuple
import numpy as np


# number of bytes tiles can be off by in comparison to max_chunk_size
COST_THRESHOLD = 800 * KB  # TODO: make smaller with better optimizers

# dtype for intermediate tile shape calculations
INTERM_DTYPE = np.dtype(np.uint32)


def _cost(tile_shape: Tuple[int, ...], dtype: np.dtype, tile_target_bytes: int, compression_factor: float) -> int:
    # TODO: docstring

    actual_bytes = np.prod(tile_shape) * dtype.itemsize // compression_factor
    abs_cost = abs(tile_target_bytes - actual_bytes)

    # higher cost for when `actual_bytes` is smaller than `target_bytes`
    if actual_bytes < tile_target_bytes:
        return abs_cost * 2

    return abs_cost


def _downscale(tile_shape: np.ndarray, sample_shape: Tuple[int, ...]) -> np.ndarray:
    # TODO: docstring
    
    return np.minimum(tile_shape, sample_shape).astype(INTERM_DTYPE)


def _propose_tile_shape(
    sample_shape: Tuple[int, ...], dtype: np.dtype, max_chunk_size: int, compression_factor: float
) -> np.ndarray:
    # TODO: docstring (use args)

    dtype_num_bytes = dtype.itemsize
    ndims = len(sample_shape)

    # TODO: factor in sample/chunk compression (ALSO DO THIS BEFORE MERGING)
    tile_target_entries = max_chunk_size // dtype_num_bytes
    tile_target_entries *= compression_factor
    tile_target_length = int(tile_target_entries ** (1.0 / ndims))

    # Tile length must be at least 16, so we don't round too close to zero
    tile_shape = max((16,) * ndims, (tile_target_length,) * ndims)

    return np.array(tile_shape, dtype=INTERM_DTYPE)


def _optimize_tile_shape(
    sample_shape: Tuple[int, ...], dtype: np.dtype, max_chunk_size: int, compression_factor: float
) -> Tuple[int, ...]:
    # TODO: docstring

    tile_shape = _propose_tile_shape(sample_shape, dtype, max_chunk_size, compression_factor)
    downscaled_tile_shape = _downscale(tile_shape, sample_shape)
    frozen_dims_mask = tile_shape != downscaled_tile_shape

    # iterate until we find a tile shape close to max chunk size
    proposed_tile_shape = downscaled_tile_shape
    while _cost(proposed_tile_shape, dtype, max_chunk_size, compression_factor) > COST_THRESHOLD:
        # TODO!
        
        break

    return tuple(proposed_tile_shape.tolist())


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
