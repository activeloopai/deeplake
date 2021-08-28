from hub.util.tiles import approximate_num_bytes, num_tiles_for_sample
from hub.core.compression import get_compression_factor
from hub.core.meta.tensor_meta import TensorMeta
from hub.constants import KB
from typing import Tuple
import numpy as np


# dtype for intermediate tile shape calculations
INTERM_DTYPE = np.dtype(np.uint32)


def _energy(tile_shape: np.ndarray, sample_shape: Tuple[int, ...], tensor_meta: TensorMeta) -> float:
    # TODO: docstring

    num_tiles = num_tiles_for_sample(tile_shape, sample_shape)
    num_bytes_per_tile = approximate_num_bytes(tile_shape, tensor_meta)
    return num_tiles * num_bytes_per_tile


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


def _perturbate_tile_shape(tile_shape: np.ndarray, unfrozen_dim_mask: np.ndarray, temperature: float) -> Tuple[int, ...]:
    # TODO: docstring

    print(temperature)


def _optimize_tile_shape(sample_shape: Tuple[int, ...], tensor_meta: TensorMeta) -> Tuple[int, ...]:
    tile_shape = _propose_tile_shape(sample_shape, tensor_meta)
    unfrozen_dim_mask = tile_shape != sample_shape

    # TODO: make params
    try_count = 0
    max_tries = 100

    best_shape = None
    lowest_energy = float("inf")

    # TODO: minimize energy with respect to tile shape
    while try_count < max_tries:
        temperature = 1 - ((try_count) / max_tries) ** 2
        _perturbate_tile_shape(tile_shape, unfrozen_dim_mask, temperature)
        energy = _energy(tile_shape, sample_shape, tensor_meta)

        if energy < lowest_energy:
            best_shape = tile_shape.copy()

        try_count += 1

    return tuple(tile_shape.tolist()), []


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
