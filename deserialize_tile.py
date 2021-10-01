from typing import Callable, Tuple
import numpy as np


def coalesce_tiles(tiles: np.ndarray, sample_shape: Tuple[int, ...]):
    # TODO: if bytes, raise error (note to deserialize)

    raise NotImplementedError


def deserialize_tiles(serialized_tiles: np.ndarray, frombytes_func: Callable[[bytes], np.ndarray]):

    raise NotImplementedError