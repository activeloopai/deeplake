import numpy as np


def array_to_bytes(array: np.ndarray) -> bytes:
    # TODO: this can be replaced with hilbert curves
    return array.tobytes()


def index_map_entry_to_bytes(map_entry: dict):
    # TODO: don't use pickle, encode `map_entry` into array
    return pickle.dumps(map_entry)
