from hub.core.compression import get_compression_factor
from hub.util.exceptions import TileOptimizerError
from hub.util.tiles import approximate_num_bytes, num_tiles_for_sample
from hub.core.meta.tensor_meta import TensorMeta
from typing import Dict, List, Tuple
import numpy as np


# dtype for intermediate tile shape calculations
INTERM_DTYPE = np.dtype(np.int32)


# this is the percentage of max steps that will allow a single dimension to be perturbated.
# if the number of iterations is less than this, then all dimensions will be perturbated at once,
# if the number of iterations is greater than this, then only a single dimension will be perturbated at a time.
# the reason is to allow better and smoother exploration of more nuanced tile shapes.
SINGLE_DIM_MIN_ITERATION_PERCENTAGE = 0.8
assert SINGLE_DIM_MIN_ITERATION_PERCENTAGE > 0 and SINGLE_DIM_MIN_ITERATION_PERCENTAGE < 1
CHANCE_FOR_SINGLE_DIM_ONLY = 0.3
assert CHANCE_FOR_SINGLE_DIM_ONLY > 0 and CHANCE_FOR_SINGLE_DIM_ONLY < 1



def _clamp(tile_shape: np.ndarray, sample_shape: Tuple[int, ...]) -> np.ndarray:
    """Clamps `tile_shape` on each dimension inclusively between 1 and sample_shape for the corresponding dimension."""

    tile_shape = np.minimum(tile_shape, sample_shape)
    return np.maximum(1, tile_shape)


def _transition_probability(
    energy_old: float, energy_new: float, temperature: float
) -> float:
    """Metropolis-Hastings acceptance function."""

    if energy_new < energy_old:
        return 1

    return np.exp(-(energy_new - energy_old) / temperature)

class TileOptimizer:

    def __init__(self, min_chunk_size: int, max_chunk_size: int, tensor_meta: TensorMeta):
        """Uses simulated annealing to find the best tile shape for a sample of a specific shape
        with respect to the tensor meta properties."""
        
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.tensor_meta = tensor_meta
        self.dtype = np.dtype(tensor_meta.dtype)

        # state
        self.compression_factor = None
        self.last_tile_shape = None
        self.last_history = None

    def average_num_bytes_per_tile(self, tile_shape: Tuple[int, ...]) -> int:
        return approximate_num_bytes(tile_shape, self.tensor_meta.dtype, self.compression_factor)

    def _energy(
        self, tile_shape: np.ndarray, sample_shape: Tuple[int, ...]
    ) -> float:
        """Function to be minimized during simulated annealing. The energy of a state is the
        cumulative waste that a tile shape has. The target tile size is `tensor_meta.max_chunk_size`.

        The distance is squared when it is not between min / max chunk size.
        """

        num_tiles = num_tiles_for_sample(tile_shape, sample_shape)
        num_bytes_per_tile = self.average_num_bytes_per_tile(tile_shape)

        distance = abs(num_bytes_per_tile - self.max_chunk_size)
        if (
            num_bytes_per_tile < self.min_chunk_size  # type: ignore
            or num_bytes_per_tile > self.max_chunk_size
        ):
            # TODO: make smoother?
            distance = distance * distance * 100

        return num_tiles * distance


    def _initial_tile_shape(
        self, sample_shape: Tuple[int, ...]
    ) -> np.ndarray:
        """Calculates the initial tile shape for simulated annealing."""

        dtype_num_bytes = self.dtype.itemsize
        ndims = len(sample_shape)

        tile_target_entries = self.max_chunk_size // dtype_num_bytes
        tile_target_entries *= self.compression_factor
        tile_target_length = int(tile_target_entries ** (1.0 / ndims))

        # Tile length must be at least 16, so we don't round too close to zero
        tile_shape = max((16,) * ndims, (tile_target_length,) * ndims)

        proposal = np.array(tile_shape, dtype=INTERM_DTYPE)
        return _clamp(proposal, sample_shape)

    
    def _perturbate_tile_shape(
        self,
        tile_shape: np.ndarray,
        sample_shape: Tuple[int, ...],
        unfrozen_dim_mask: np.ndarray,
        allow_single_dim_only: bool,
        max_magnitude: int = 1000,
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

        new_tile_shape = tile_shape.copy()

        delta = np.zeros(new_tile_shape.shape, dtype=INTERM_DTYPE)
        delta_view = delta[unfrozen_dim_mask]

        random_magnitude = np.random.uniform(-max_magnitude, max_magnitude + 1)
        if random_magnitude == 0:
            random_magnitude = 1

        if allow_single_dim_only and np.random.uniform() < CHANCE_FOR_SINGLE_DIM_ONLY:
            # isolate a single dimension to perturbate, do so with a magnitude of 1
            i = np.random.choice(range(delta_view.size))
            random_magnitude = abs(random_magnitude * self.temperature)
            delta_view[i] = np.random.choice([-random_magnitude, random_magnitude])
        else:
            # all dims should have the same delta
            delta_view[:] = random_magnitude

        new_tile_shape[unfrozen_dim_mask] += delta_view
        new_tile_shape = _clamp(new_tile_shape, sample_shape)

        if np.all(new_tile_shape == tile_shape):
            return self._perturbate_tile_shape(tile_shape, sample_shape, unfrozen_dim_mask, allow_single_dim_only, max_magnitude=max_magnitude)

        return new_tile_shape


    def _anneal_tile_shape(
        self, sample_shape: Tuple[int, ...], max_iterations: int = 2000
    ) -> Tuple[Tuple[int, ...], List[Dict]]:
        """Use simulated annealing to find a tile shape that is between the min / max chunk size of `tensor_meta`.

        Simulated Annealing:
            https://en.wikipedia.org/wiki/Simulated_annealing
            State variable: `tile_shape`

        Args:
            sample_shape (Tuple[int]): Shape of the sample that is being tiled.
            max_iterations (int): Maximum number of simulated annealing steps.

        Returns:
            Tuple[Tuple[int], List[Dict]]: Tile shape and history of simulated annealing.
        """

        min_single_dim_iterations = int(SINGLE_DIM_MIN_ITERATION_PERCENTAGE * max_iterations)

        tile_shape = self._initial_tile_shape(sample_shape)
        unfrozen_dim_mask = np.ones(len(tile_shape), dtype=bool) # tile_shape != sample_shape

        self.current_iteration = 0
        best_shape = tile_shape
        lowest_energy = self._energy(tile_shape, sample_shape)
        history = []

        # minimize energy with respect to tile shape
        while self.current_iteration < max_iterations:
            allow_single_dim_only = self.current_iteration > min_single_dim_iterations

            self.temperature = 1 - ((self.current_iteration) / max_iterations) ** 2
            new_tile_shape = self._perturbate_tile_shape(
                tile_shape, sample_shape, unfrozen_dim_mask, allow_single_dim_only
            )

            energy = self._energy(tile_shape, sample_shape)
            new_energy = self._energy(new_tile_shape, sample_shape)

            p = _transition_probability(energy, new_energy, self.temperature)
            r = np.random.uniform()

            if p > r:
                tile_shape = new_tile_shape
                if new_energy < lowest_energy:
                    best_shape = new_tile_shape.copy()
                    lowest_energy = new_energy

            history.append({"old_energy": energy, "new_energy": new_energy, "old_shape": tile_shape, "new_shape": new_tile_shape})

            self.current_iteration += 1

        return tuple(best_shape.tolist()), history  # type: ignore


    def _validate_tile_shape(
        self, tile_shape: Tuple[int, ...], sample_shape: Tuple[int, ...]
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
        num_tiles = num_tiles_for_sample(tile_shape, sample_shape)
        num_bytes_per_tile = self.average_num_bytes_per_tile(tile_shape)

        failure_reason = None
        if num_tiles <= 1:
            failure_reason = f"Number of tiles should be > 1. Got {num_tiles}"
        elif not np.all(tile_shape_arr <= sample_shape):
            failure_reason = "Tile shape must not be > sample shape on any dimension"
        elif np.any(tile_shape_arr <= 0):
            failure_reason = "Tile shape must not be <= 0 on any dimension"
        elif num_bytes_per_tile > self.max_chunk_size:
            failure_reason = f"Number of bytes per tile ({num_bytes_per_tile}) is larger than what is allowed ({self.max_chunk_size})"
        elif num_bytes_per_tile < self.min_chunk_size:
            failure_reason = f"Number of bytes per tile ({num_bytes_per_tile}) is smaller than what is allowed ({self.min_chunk_size})"

        if failure_reason is not None:
            failure_reason += self.history_subset_str()

            raise TileOptimizerError(failure_reason, tile_shape, sample_shape)

    
    def history_subset_str(self):
        s = f"\nHistory:"

        # add history to failure reason
        for step in self.last_history[-50:]:
            s += f"\n{step}"

        return s


    def optimize(
        self,
        sample_shape: Tuple[int, ...],
        compression_factor: int=None,
        validate: bool = True,
        return_history: bool = False,
    ) -> Tuple[int, ...]:
        """Find a tile shape using simulated annealing.
    
        Args:
            sample_shape (Tuple[int]): Shape of the sample that is being tiled.
            validate (bool): Whether to validate the tile shape.
            return_history (bool): Whether to return the history of the simulated annealing. Useful for debugging.
    
        Returns:
            The tile shape found by simulated annealing.
        """

        self.compression_factor = compression_factor or get_compression_factor(self.tensor_meta)
    
        tile_shape, history = self._anneal_tile_shape(sample_shape)
    
        self.last_tile_shape = tile_shape
        self.last_history = history

        if validate:
            self._validate_tile_shape(tile_shape, sample_shape)

        self.compression_factor = None
    
        if return_history:
            return tile_shape, history  # type: ignore
    
        return tile_shape
