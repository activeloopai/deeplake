import numpy as np


def array_to_bytes(array: np.ndarray) -> bytes:
    # TODO: this can be replaced with hilbert curves
    return array.tobytes()


def index_map_entry_to_bytes(map_entry: dict):
    # TODO: don't use pickle, encode `map_entry` into array
    return pickle.dumps(map_entry)


def normalize_and_batchify_shape(array: np.ndarray, batched: bool) -> np.ndarray:
    # if the first axis is of length 1, but batched is true, it is only a single sample & squeeze will remove it
    actually_batched = batched and array.shape[0] != 1
    array = array.squeeze()
    if not actually_batched:
        array = np.expand_dims(array, axis=0)
    if len(array.shape) == 1:
        array = np.expand_dims(array, axis=0)
    return array
