from hub.util.exceptions import TileOptimizerError
from hub.util.tiles import approximate_num_bytes, num_tiles_for_sample
from hub.core.compression import get_compression_factor
from hub.core.meta.tensor_meta import TensorMeta
from typing import Dict, List, Tuple
import numpy as np


# dtype for intermediate tile shape calculations
INTERM_DTYPE = np.dtype(np.int32)


def _energy(
    tile_shape: np.ndarray, sample_shape: Tuple[int, ...], tensor_meta: TensorMeta
) -> float:
    """Function to be minimized during simulated annealing. The energy of a state is the
    cumulative waste that a tile shape has. The target tile size is `tensor_meta.max_chunk_size`.

    The distance is squared when it is not between min / max chunk size.
    """

    num_tiles = num_tiles_for_sample(tile_shape, sample_shape)
    num_bytes_per_tile = approximate_num_bytes(tile_shape, tensor_meta)

    distance = abs(num_bytes_per_tile - tensor_meta.max_chunk_size)
    if (
        num_bytes_per_tile < tensor_meta.min_chunk_size  # type: ignore
        or num_bytes_per_tile > tensor_meta.max_chunk_size
    ):
        distance = distance * distance

    return num_tiles * distance


def _clamp(tile_shape: np.ndarray, sample_shape: Tuple[int, ...]) -> np.ndarray:
    """Clamps `tile_shape` on each dimension inclusively between 1 and sample_shape for the corresponding dimension."""

    tile_shape = np.minimum(tile_shape, sample_shape)
    return np.maximum(1, tile_shape)


def _initial_tile_shape(
    sample_shape: Tuple[int, ...], tensor_meta: TensorMeta
) -> np.ndarray:
    """Calculates the initial tile shape for simulated annealing."""

    dtype_num_bytes = np.dtype(tensor_meta.dtype).itemsize
    ndims = len(sample_shape)

    tile_target_entries = tensor_meta.max_chunk_size // dtype_num_bytes
    tile_target_entries *= get_compression_factor(tensor_meta)
    tile_target_length = int(tile_target_entries ** (1.0 / ndims))

    # Tile length must be at least 16, so we don't round too close to zero
    tile_shape = max((16,) * ndims, (tile_target_length,) * ndims)

    proposal = np.array(tile_shape, dtype=INTERM_DTYPE)
    return _clamp(proposal, sample_shape)


def _perturbate_tile_shape(
    tile_shape: np.ndarray,
    sample_shape: Tuple[int, ...],
    unfrozen_dim_mask: np.ndarray,
    max_magnitude: int = 100,
) -> np.ndarray:
    """Pertubrate the tile shape in a random direction, only where `unfrozen_dim_mask` is True.

    Args:
        tile_shape (np.ndarray): Tile shape to perturbate.
        sample_shape (Tuple[int]): Shape of the sample that is being tiled.
        unfrozen_dim_mask (np.ndarray): Boolean mask of dimensions that are not frozen.
        max_magnitude (int): No dimension will be perturbated by more than this value
            in the negative or positive direction.

    Returns:
        A new numpy array representing the perturbated tile shape.
    """

    num_unfrozen_dims = len(unfrozen_dim_mask.shape)

    new_tile_shape = tile_shape.copy()
    new_tile_shape[unfrozen_dim_mask] += (
        np.random.uniform(-max_magnitude, max_magnitude + 1, size=num_unfrozen_dims)
    ).astype(INTERM_DTYPE)

    return _clamp(new_tile_shape, sample_shape)


def _transition_probability(
    energy_old: float, energy_new: float, temperature: float
) -> float:
    """Metropolis-Hastings acceptance function."""

    if energy_new < energy_old:
        return 1

    return np.exp(-(energy_new - energy_old) / temperature)


def anneal_tile_shape(
    sample_shape: Tuple[int, ...], tensor_meta: TensorMeta, max_steps: int = 1000
) -> Tuple[Tuple[int, ...], List[Dict]]:
    """Use simulated annealing to find a tile shape that is between the min / max chunk size of `tensor_meta`.

    Simulated Annealing:
        https://en.wikipedia.org/wiki/Simulated_annealing
        State variable: `tile_shape`

    Args:
        sample_shape (Tuple[int]): Shape of the sample that is being tiled.
        tensor_meta (TensorMeta): Tensor meta for the tensor that is being tiled. Should include
            `min_chunk_size` and `max_chunk_size`.
        max_steps (int): Maximum number of simulated annealing steps.

    Returns:
        Tuple[Tuple[int], List[Dict]]: Tile shape and history of simulated annealing.
    """

    tile_shape = _initial_tile_shape(sample_shape, tensor_meta)
    unfrozen_dim_mask = tile_shape != sample_shape

    try_count = 0
    best_shape = None
    lowest_energy = float("inf")
    history = []

    # minimize energy with respect to tile shape
    while try_count < max_steps:
        temperature = 1 - ((try_count) / max_steps) ** 2
        new_tile_shape = _perturbate_tile_shape(
            tile_shape, sample_shape, unfrozen_dim_mask
        )

        energy = _energy(tile_shape, sample_shape, tensor_meta)
        new_energy = _energy(new_tile_shape, sample_shape, tensor_meta)

        p = _transition_probability(energy, new_energy, temperature)
        r = np.random.uniform()

        if p > r:
            tile_shape = new_tile_shape
            if new_energy < lowest_energy:
                best_shape = new_tile_shape.copy()
                lowest_energy = new_energy
            history.append({"energy": new_energy})

        try_count += 1

    return tuple(best_shape.tolist()), history  # type: ignore


def _validate_tile_shape(
    tile_shape: Tuple[int, ...], sample_shape: Tuple[int, ...], tensor_meta: TensorMeta
):
    """Raises appropriate errors if the tile shape is not valid.

    Args:
        tile_shape (Tuple[int, ...]): Tile shape to validate.
        sample_shape (Tuple[int, ...]): Shape of the sample that is being tiled.
        tensor_meta (TensorMeta): Tile meta for the tensor that is being tiled.

    Raises:
        TileOptimizerError: If the tile shape is not valid.
    """

    tile_shape_arr = np.array(tile_shape)
    average_num_bytes_per_tile = approximate_num_bytes(tile_shape_arr, tensor_meta)

    failure_reason = None
    if not np.all(tile_shape_arr <= sample_shape):
        failure_reason = "Tile shape must not be > sample shape on any dimension"
    elif np.any(tile_shape_arr <= 0):
        failure_reason = "Tile shape must not be <= 0 on any dimension"
    elif average_num_bytes_per_tile > tensor_meta.max_chunk_size:  # type: ignore
        failure_reason = f"Number of bytes per tile ({average_num_bytes_per_tile}) is larger than what is allowed ({tensor_meta.max_chunk_size})"  # type: ignore
    elif average_num_bytes_per_tile < tensor_meta.min_chunk_size:  # type: ignore
        failure_reason = f"Number of bytes per tile ({average_num_bytes_per_tile}) is smaller than what is allowed ({tensor_meta.min_chunk_size})"  # type: ignore

    if failure_reason is not None:
        raise TileOptimizerError(failure_reason, tile_shape, sample_shape)


def optimize_tile_shape(
    sample_shape: Tuple[int, ...],
    tensor_meta: TensorMeta,
    validate: bool = True,
    return_history: bool = False,
) -> Tuple[int, ...]:
    """Find a tile shape using simulated annealing.

    Args:
        sample_shape (Tuple[int]): Shape of the sample that is being tiled.
        tensor_meta (TensorMeta): Tensor meta for the tensor that is being tiled.
        validate (bool): Whether to validate the tile shape.
        return_history (bool): Whether to return the history of the simulated annealing. Useful for debugging.

    Returns:
        The tile shape found by simulated annealing.
    """

    tile_shape, history = anneal_tile_shape(sample_shape, tensor_meta)

    if validate:
        _validate_tile_shape(tile_shape, sample_shape, tensor_meta)

    if return_history:
        return tile_shape, history  # type: ignore

    return tile_shape
