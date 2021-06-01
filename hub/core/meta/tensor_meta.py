import numpy as np
import pickle

from hub.core.typing import StorageProvider
from hub.constants import DEFAULT_CHUNK_SIZE
from hub.util.keys import get_tensor_meta_key
from hub.util.array import normalize_and_batchify_shape


def write_tensor_meta(key: str, storage: StorageProvider, meta: dict):
    _validate_tensor_meta(meta)
    storage[get_tensor_meta_key(key)] = pickle.dumps(meta)

def read_tensor_meta(key: str, storage: StorageProvider) -> dict:
    return pickle.loads(storage[get_tensor_meta_key(key)])

def tensor_meta_from_array(
    array: np.ndarray, batched: bool, chunk_size: int = DEFAULT_CHUNK_SIZE
):
    array = normalize_and_batchify_shape(array, batched=batched)

    tensor_meta = {
        "chunk_size": chunk_size,
        "dtype": array.dtype.name,
        "length": 0,
        "min_shape": tuple(array.shape[1:]),
        "max_shape": tuple(array.shape[1:]),
        # TODO: add entry in meta for which tobytes function is used and handle mismatch versions for this
    }

    return tensor_meta

def _validate_tensor_meta(meta: dict):
    # TODO
    pass