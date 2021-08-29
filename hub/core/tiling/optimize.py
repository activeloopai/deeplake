from hub.util.tiles import approximate_num_bytes, num_tiles_for_sample
from hub.core.compression import get_compression_factor
from hub.core.meta.tensor_meta import TensorMeta
from typing import Tuple
import numpy as np


# dtype for intermediate tile shape calculations
INTERM_DTYPE = np.dtype(np.int32)


def _energy(tile_shape: np.ndarray, sample_shape: Tuple[int, ...], tensor_meta: TensorMeta) -> float:
    # TODO: docstring

    num_tiles = num_tiles_for_sample(tile_shape, sample_shape)
    num_bytes_per_tile = approximate_num_bytes(tile_shape, tensor_meta)

    distance = abs(num_bytes_per_tile - tensor_meta.max_chunk_size)
    if num_bytes_per_tile < tensor_meta.min_chunk_size or num_bytes_per_tile > tensor_meta.max_chunk_size:
        distance = distance * distance
    
    return num_tiles * distance


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


def _perturbate_tile_shape(tile_shape: np.ndarray, sample_shape: Tuple[int, ...], unfrozen_dim_mask: np.ndarray, max_magnitude: int=100) -> Tuple[int, ...]:
    # TODO: docstring

    num_unfrozen_dims = len(unfrozen_dim_mask.shape)

    new_tile_shape = tile_shape.copy()
    new_tile_shape[unfrozen_dim_mask] += (np.random.uniform(-max_magnitude, max_magnitude+1, size=num_unfrozen_dims)).astype(INTERM_DTYPE)

    return _clamp(new_tile_shape, sample_shape)


def _transition_probability(energy_old: float, energy_new: float, temperature: float) -> float:
    if energy_new < energy_old:
        return 1

    return np.exp(-(energy_new - energy_old) / temperature)


def _optimize_tile_shape(sample_shape: Tuple[int, ...], tensor_meta: TensorMeta) -> Tuple[int, ...]:
    tile_shape = _propose_tile_shape(sample_shape, tensor_meta)
    unfrozen_dim_mask = tile_shape != sample_shape

    # TODO: make params
    try_count = 0
    max_tries = 1000

    best_shape = None
    lowest_energy = float("inf")

    history = []

    # TODO: minimize energy with respect to tile shape
    while try_count < max_tries:
        temperature = 1 - ((try_count) / max_tries) ** 2
        new_tile_shape = _perturbate_tile_shape(tile_shape, sample_shape, unfrozen_dim_mask)

        energy = _energy(tile_shape, sample_shape, tensor_meta)
        new_energy = _energy(new_tile_shape, sample_shape, tensor_meta)

        if _transition_probability(energy, new_energy, temperature) > np.random.uniform():
            tile_shape = new_tile_shape
            if new_energy < lowest_energy:
                best_shape = new_tile_shape.copy()
                lowest_energy = new_energy

            history.append({"energy": new_energy})
        try_count += 1

    return tuple(best_shape.tolist()), history


def _validate_tile_shape(
    tile_shape: Tuple[int, ...], sample_shape: Tuple[int, ...], tensor_meta: TensorMeta
):
    # TODO: docstring

    tile_shape = np.array(tile_shape)

    if not np.all(tile_shape <= sample_shape):
        raise Exception(f"Invalid tile shape {tile_shape} for sample shape {sample_shape}.")  # TODO

    if np.any(tile_shape <= 0):
        raise Exception()

    # TODO: exceptions.py
    average_num_bytes_per_tile = approximate_num_bytes(tile_shape, tensor_meta)
    if average_num_bytes_per_tile > tensor_meta.max_chunk_size:
        raise Exception(f"Average num bytes per tile {average_num_bytes_per_tile} is greater than max chunk size {tensor_meta.max_chunk_size}.")
    elif average_num_bytes_per_tile < tensor_meta.min_chunk_size:
        raise Exception(f"Average num bytes per tile {average_num_bytes_per_tile} is less than min chunk size {tensor_meta.min_chunk_size}.") 

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
