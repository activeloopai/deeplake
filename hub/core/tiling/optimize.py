from hub.util.tiles import approximate_num_bytes
from hub.core.compression import get_compression_factor
from hub.core.meta.tensor_meta import TensorMeta
from hub.constants import KB
from typing import Tuple
import numpy as np


# number of simulated annealing iterations
ANNEALING_MAX_TRIES = 100

# rectangularization is when the proposed tile shape is no longer square (all dimensions are not equal)
ANNEALING_CHANCE_TO_RECTANGULARIZE = 0.01
ANNEALING_MIN_TRIES_TO_RECTANGULARIZE = 5

ANNEALING_MAX_MAGNITUDE = 10

# number of bytes tiles can be off by in comparison to max_chunk_size
COST_THRESHOLD = 800 * KB  # TODO: make smaller with better optimizers

# dtype for intermediate tile shape calculations
INTERM_DTYPE = np.dtype(np.uint32)



def _cost(tile_shape: Tuple[int, ...], tensor_meta: TensorMeta) -> int:
    # TODO: docstring

    actual_bytes = approximate_num_bytes(tile_shape, tensor_meta)
    abs_cost = abs(tensor_meta.max_chunk_size - actual_bytes)

    # higher cost if the actual bytes is less than the min chunk size
    if actual_bytes < tensor_meta.min_chunk_size:
        return abs_cost * 2

    return abs_cost


def _downscale(tile_shape: np.ndarray, sample_shape: Tuple[int, ...]) -> np.ndarray:
    # TODO: docstring

    return np.minimum(tile_shape, sample_shape).astype(INTERM_DTYPE)


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

    return np.array(tile_shape, dtype=INTERM_DTYPE)


def _optimize_tile_shape(
    sample_shape: Tuple[int, ...], tensor_meta: TensorMeta
) -> Tuple[int, ...]:
    # TODO: docstring
    # TODO: explain using simulated annealing

    tile_shape = _propose_tile_shape(sample_shape, tensor_meta)
    downscaled_tile_shape = _downscale(tile_shape, sample_shape)
    unfrozen_dims_mask = tile_shape == downscaled_tile_shape

    # it is expected that at least 1 dimension is unfrozen. this shouldn't happen, but in the event it does, this
    # check will save a lot of debugging time
    if np.all(unfrozen_dims_mask == False):
        raise NotImplementedError(f"Cannot determine a tile shape when all axes are too small. Sample shape: {sample_shape}.")

    target_chunk_size = tensor_meta.max_chunk_size
    tries = 0
    num_dims = len(sample_shape)


    # state of the optimizer
    proposed_tile_shape = downscaled_tile_shape  
    best_tile_shape = None
    best_cost = float("inf")

    history = []

    get_cost = lambda: _cost(proposed_tile_shape, tensor_meta)
    is_under_cost_threshold = lambda: get_cost() < COST_THRESHOLD
    is_under_max_tries = lambda: tries < ANNEALING_MAX_TRIES
    calculate_temperature = lambda: (1 - (tries + 1) / ANNEALING_MAX_TRIES)
    random_magnitude = lambda: np.random.randint(1, int(max(ANNEALING_MAX_MAGNITUDE * calculate_temperature(), 2)))
    track_history = lambda: history.append({
        "cost": get_cost(), 
        "tile_shape": proposed_tile_shape, 
        "temperature": calculate_temperature(),
    })
    can_rectangularize = lambda: tries > ANNEALING_MIN_TRIES_TO_RECTANGULARIZE
    should_rectangularize = lambda: np.random.random() < ANNEALING_CHANCE_TO_RECTANGULARIZE

    def set_best_tile_shape():
        nonlocal best_tile_shape, best_cost

        cost = get_cost()
        if cost < best_cost:
            best_cost = cost
            best_tile_shape = proposed_tile_shape


    track_history()
    set_best_tile_shape()

    # iterate until we find a tile shape close to max chunk size
    while is_under_cost_threshold() and is_under_max_tries():
        average_tile_size = approximate_num_bytes(proposed_tile_shape, tensor_meta)

        sign = 1 if average_tile_size < target_chunk_size else -1
        magnitude = random_magnitude() * sign

        # TODO: explain this
        if can_rectangularize() and should_rectangularize():
            random_index = np.random.choice(np.arange(num_dims))
            grad = np.zeros(num_dims)
            grad[random_index] = magnitude
        else:
            grad = np.ones(num_dims) * magnitude

        proposed_tile_shape += grad.astype(INTERM_DTYPE)
        print(grad)
        
        # float_tile_shape = proposed_tile_shape.astype(np.float64)
        # print((float_tile_shape[unfrozen_dims_mask] * grad).astype(INTERM_DTYPE))
        # proposed_tile_shape = (float_tile_shape[unfrozen_dims_mask] * grad).astype(INTERM_DTYPE)
        
        tries += 1

        set_best_tile_shape()
        track_history()

    # return tuple(proposed_tile_shape.tolist()), history
    return tuple(best_tile_shape.tolist()), history


def _validate_tile_shape(
    tile_shape: Tuple[int, ...], sample_shape: Tuple[int, ...], tensor_meta: TensorMeta
):
    # TODO: docstring

    if not np.all(np.array(tile_shape) >= sample_shape):
        raise Exception()  # TODO

    cost = _cost(tile_shape, tensor_meta)
    if cost > COST_THRESHOLD:
        raise Exception(
            f"Cost too large ({cost}) for tile shape {tile_shape} and max chunk size {tensor_meta.max_chunk_size}. Dtype={tensor_meta.dtype}"
        )  # TODO: exceptions.py


def optimize_tile_shape(sample_shape: Tuple[int, ...], tensor_meta: TensorMeta, validate: bool=True, return_history: bool=False):
    # TODO: docstring
    
    tile_shape, history = _optimize_tile_shape(sample_shape, tensor_meta)

    if validate:
        _validate_tile_shape(tile_shape, sample_shape, tensor_meta)

    if return_history:
        return tile_shape, history

    return tile_shape
