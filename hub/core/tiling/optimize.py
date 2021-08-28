from hub.util.tiles import approximate_num_bytes, num_tiles_for_sample
from hub.core.compression import get_compression_factor
from hub.core.meta.tensor_meta import TensorMeta
from hub.constants import KB
from typing import Tuple
import numpy as np


# dtype for intermediate tile shape calculations
INTERM_DTYPE = np.dtype(np.uint32)



def _clamp(tile_shape: np.ndarray, sample_shape: Tuple[int, ...]) -> np.ndarray:
    # TODO: docstring

    tile_shape = np.minimum(tile_shape, sample_shape)
    return np.maximum(1, tile_shape)

def _propose_tile_shape(
    sample_shape: Tuple[int, ...], tensor_meta: TensorMeta
) -> np.ndarray:
    # TODO: docstring (use args)

    dtype_num_bytes = np.dtype(tensor_meta.dtype).itemsize
    ndims = len(sample_shape)

    tile_target_entries = tensor_meta.max_chunk_size // dtype_num_bytes
    tile_target_entries *= get_compression_factor(tensor_meta)
    tile_target_length = int(tile_target_entries ** (1.0 / ndims))

    # Tile length must be at least 16, so we don't round too close to zero
    tile_shape = max((16,) * ndims, (tile_target_length,) * ndims)

    proposal = np.array(tile_shape, dtype=INTERM_DTYPE)
    return _clamp(proposal, sample_shape)


def _optimize_tile_shape(sample_shape: Tuple[int, ...], tensor_meta: TensorMeta) -> Tuple[int, ...]:
    tile_shape = _propose_tile_shape(sample_shape, tensor_meta)
    return tile_shape, []


def _validate_tile_shape(
    tile_shape: Tuple[int, ...], sample_shape: Tuple[int, ...], tensor_meta: TensorMeta
):
    # TODO: docstring

    tile_shape = np.array(tile_shape)

    if not np.all(tile_shape <= sample_shape):
        raise Exception(f"Invalid tile shape {tile_shape} for sample shape {sample_shape}.")  # TODO

    if np.any(tile_shape <= 0):
        raise Exception()

    # TODO: uncomment
    # cost = _cost(tile_shape, sample_shape, tensor_meta)
    # if cost > COST_THRESHOLD:
    #     raise Exception(
    #         f"Cost too large ({cost}) for tile shape {tile_shape} and max chunk size {tensor_meta.max_chunk_size}. Dtype={tensor_meta.dtype}"
    #     )  # TODO: exceptions.py


def optimize_tile_shape(sample_shape: Tuple[int, ...], tensor_meta: TensorMeta, validate: bool=True, return_history: bool=False):
    # TODO: docstring
    
    tile_shape, history = _optimize_tile_shape(sample_shape, tensor_meta)

    if validate:
        _validate_tile_shape(tile_shape, sample_shape, tensor_meta)

    if return_history:
        return tile_shape, history

    return tile_shape
