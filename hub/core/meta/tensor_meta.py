from hub.util.exceptions import TensorMetaInvalidValue, TensorMetaMissingKey
import numpy as np
import pickle # TODO: NEVER USE PICKLE
from typing import Any, Callable

from hub.core.typing import StorageProvider
from hub.constants import DEFAULT_CHUNK_SIZE
from hub.util.keys import get_tensor_meta_key
from hub.util.array import normalize_and_batchify_shape


def write_tensor_meta(key: str, storage: StorageProvider, meta: dict):
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

def validate_tensor_meta(meta: dict):
    _raise_if_no_key("chunk_size", meta)
    _raise_if_condition("chunk_size", meta, lambda chunk_size: chunk_size <= 0, "Chunk size must be greater than 0.")

    _raise_if_no_key("dtype", meta)
    dtype_type = type(meta["dtype"])
    if dtype_type == str:
        _raise_if_condition("dtype", meta, lambda dtype: not _is_dtype_supported_by_numpy(dtype), \
            "Datatype must be supported by numpy. List of available numpy dtypes found here: https://numpy.org/doc/stable/user/basics.types.html")
    else:
        _raise_if_condition("dtype", meta, lambda dtype: type(dtype) != np.dtype, \
            "Datatype must be of type string or numpy.dtype.")


def _raise_if_no_key(key: str, meta: dict):
    if key not in meta:
        raise TensorMetaMissingKey(key)

    
def _raise_if_condition(key: str, meta: dict, condition: Callable[[Any], bool], explanation: str=""):
    v = meta[key]
    if condition(v):
        raise TensorMetaInvalidValue(key, v, explanation)


def _is_dtype_supported_by_numpy(dtype: str):
    try:
        np.dtype(dtype)
        return True
    except:
        return False