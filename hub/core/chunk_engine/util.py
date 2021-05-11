import os
import numpy as np
from hub import constants


def array_to_bytes(array: np.ndarray) -> bytes:
    return array.tobytes()


def normalize_and_batchify_shape(array: np.ndarray, batched: bool) -> np.ndarray:
    """Remove all `array` axes with a length of 1 & add a batch dimension if needed.

    Args:
        array (np.ndarray): Array to be normalized/batchified.
        batched (bool): If True, `array.shape[0]` is assumed to be the batch axis. If False,
            an axis will be added such that `array.shape[0] == 1`.

    Returns:
        np.ndarray: Array with a batch axis and all other axis sizes are > 1.
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
