from hub.core.compression import get_compression_factor
from hub.util.exceptions import TileOptimizerError
from hub.util.tiles import approximate_num_bytes, num_tiles_for_sample
from hub.core.meta.tensor_meta import TensorMeta
from typing import Dict, List, Tuple
import numpy as np


# dtype for intermediate tile shape calculations
INTERM_DTYPE = np.dtype(np.int32)



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

        distance = ((np.average([self.min_chunk_size, self.max_chunk_size])) - (num_bytes_per_tile * num_tiles))

        if num_bytes_per_tile < self.min_chunk_size:
            distance = distance * distance
        elif num_bytes_per_tile > self.max_chunk_size:
            distance = distance * distance

        return distance


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


    def _random_delta(self, tile_shape: np.ndarray, sample_shape: Tuple[int, ...], use_temperature: bool=True) -> int:
        # TODO: docstring

        assert self.num_unfrozen_dims > 0

        mask = self.unfrozen_dim_mask

        assert len(tile_shape) == len(sample_shape)

        delta = np.zeros(self.num_unfrozen_dims, dtype=INTERM_DTYPE)

        i = 0
        for tile_shape_dim in np.nditer(tile_shape):
            # don't perturbate frozen dims
            if not mask[i]:
                continue

            sample_shape_dim = sample_shape[i]

            can_perturbate_negative = tile_shape_dim > 1
            can_perturbate_positive = tile_shape_dim < sample_shape_dim

            if not can_perturbate_negative and not can_perturbate_positive:
                raise Exception

            low_diff = tile_shape_dim - 1
            high_diff = sample_shape_dim - tile_shape_dim

            if use_temperature:
                low_diff = int(max(1, low_diff * self.temperature)) + 1
                high_diff = int(max(1, low_diff * self.temperature)) + 1
        
            # TODO: can inject a gradient here with `p=` to bias towards low energy
            sign = np.random.choice([-1, 1])

            if can_perturbate_negative and can_perturbate_positive:
                if sign == -1:
                    delta[i] = -np.random.randint(1, low_diff)
                else:
                    delta[i] = np.random.randint(1, high_diff)
            elif can_perturbate_negative:
                delta[i] = -np.random.randint(1, low_diff)
            else:
                delta[i] = np.random.randint(1, high_diff)

            i += 1

        # p is a hyperparameter (can even be predicted by a model)
        # 0=non-uniform, 1=uniform, 2=isolate
        action = np.random.choice([0, 1, 2], p=[0.1, 0.5, 0.4])

        if action == 0:
            # all dimensions may be non-uniform
            pass
        elif action == 1:
            # all dimensions are a uniform value
            value = np.random.choice(delta)
            delta[:] = value
        elif action == 2:
            # only a single dimension has a non-zero delta
            i = np.random.choice(range(delta.size))
            temp = delta[i]
            delta[:] = 0
            delta[i] = temp

        print(delta)
            
        return delta

    
    def _perturbate_tile_shape(
        self,
        tile_shape: np.ndarray,
        sample_shape: Tuple[int, ...],
    ) -> np.ndarray:
        """Pertubrate the tile shape in a random direction, only where `unfrozen_dim_mask` is True.

        Args:
            tile_shape (np.ndarray): Tile shape to perturbate.
            sample_shape (Tuple[int]): Shape of the sample that is being tiled.

        Returns:
            A new numpy array representing the perturbated tile shape.
        """

        mask = self.unfrozen_dim_mask

        new_tile_shape = tile_shape.copy()

        delta = self._random_delta(tile_shape, sample_shape, use_temperature=True)
        print(delta)
        new_tile_shape[mask] += delta
        new_tile_shape = _clamp(new_tile_shape, sample_shape)

        if np.all(new_tile_shape == tile_shape):
            return self._perturbate_tile_shape(tile_shape, sample_shape)

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

        self.max_iterations = max_iterations


        tile_shape = self._initial_tile_shape(sample_shape)
        self.unfrozen_dim_mask = tile_shape != sample_shape
        self.num_unfrozen_dims = np.sum(self.unfrozen_dim_mask)

        self.current_iteration = 0
        best_shape = tile_shape
        lowest_energy = self._energy(tile_shape, sample_shape)
        history = []

        # minimize energy with respect to tile shape
        while self.current_iteration < max_iterations:
            self.temperature = 1 - ((self.current_iteration) / max_iterations) ** 2
            new_tile_shape = self._perturbate_tile_shape(
                tile_shape, sample_shape
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

        print(self.history_subset_str())

        if validate:
            self._validate_tile_shape(tile_shape, sample_shape)

        self.compression_factor = None
    
        if return_history:
            return tile_shape, history  # type: ignore
    
        return tile_shape
