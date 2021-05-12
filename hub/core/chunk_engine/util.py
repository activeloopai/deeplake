import os
import numpy as np
from hub import constants

from typing import Tuple


def array_to_bytes(array: np.ndarray) -> bytes:
    return array.tobytes()


def get_random_array(shape: Tuple[int], dtype: str) -> np.ndarray:
    dtype = dtype.lower()

    if "int" in dtype:
        low = np.iinfo(dtype).min
        high = np.iinfo(dtype).max
        return np.random.randint(low=low, high=high, size=shape, dtype=dtype)

    if "float" in dtype:
        # get float16 because np.random.uniform doesn't support the `dtype` argument.
        low = np.finfo("float16").min
        high = np.finfo("float16").max
        return np.random.uniform(low=low, high=high, size=shape).astype(dtype)

    if "bool" in dtype:
        a = np.random.uniform(size=shape)
        return a > 0.5

    raise ValueError("Dtype %s not supported." % dtype)


def normalize_and_batchify_shape(array: np.ndarray, batched: bool) -> np.ndarray:
    """Remove all `array.shape` axes with a value of 1 & add a batch dimension if needed.

    Example 1:
        input_array.shape = (10, 1, 5)
        batched = False
        output_array.shape = (1, 10, 5)  # batch axis is added

    Example 2:
        input_array.shape = (1, 100, 1, 1, 3)
        batched = True
        output_array.shape = (1, 100, 3)  # batch axis is preserved

    Args:
        array (np.ndarray): Array that will have it's shape normalized/batchified.
        batched (bool): If True, `array.shape[0]` is assumed to be the batch axis. If False,
            an axis will be added such that `array.shape[0] == 1`.

    Returns:
        np.ndarray: Array with a guarenteed batch dimension. `out_array.shape[1:]` will always be > 1.
            `out_array.shape[0]` may be >= 1.
    """

    # if the first axis is of length 1, but batched is true, it is only a single sample & squeeze will remove it
    actually_batched = batched and array.shape[0] != 1
    array = array.squeeze()
    if not actually_batched:
        array = np.expand_dims(array, axis=0)
    if len(array.shape) == 1:
        array = np.expand_dims(array, axis=0)
    return array


def get_chunk_key(key: str, chunk_name: str) -> str:
    return os.path.join(key, constants.CHUNKS_FOLDER, chunk_name)


def get_meta_key(key: str) -> str:
    return os.path.join(key, constants.META_FILENAME)


def get_index_map_key(key: str) -> str:
    return os.path.join(key, constants.INDEX_MAP_FILENAME)
