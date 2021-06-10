import pickle  # TODO: NEVER USE PICKLE
from typing import Any, Callable, Optional

import numpy as np

from hub.constants import DEFAULT_CHUNK_SIZE, DEFAULT_DTYPE, DEFAULT_HTYPE
from hub.core.typing import StorageProvider
from hub.util.exceptions import TensorMetaInvalidValue, TensorMetaMissingKey
from hub.util.keys import get_tensor_meta_key
from hub.util.array import normalize_and_batchify_array_shape

"""----"""
from hub.core.meta.meta import CallbackDict, CallbackList, Meta
from hub.core.typing import StorageProvider


HTYPE_PROPERTIES = {
    DEFAULT_HTYPE: {"dtype": DEFAULT_DTYPE},
    "image": {"dtype": "uint8"},
}


def _required_meta_from_htype(htype: str) -> dict:
    if htype not in HTYPE_PROPERTIES:
        raise Exception()  # TODO: exceptions.py

    defaults = HTYPE_PROPERTIES[htype]

    required_meta = {
        "htype": htype,
        "chunk_size": DEFAULT_CHUNK_SIZE,
    }

    required_meta.update(defaults)
    return required_meta



def create_tensor_meta(key: str, storage: StorageProvider, htype: str=DEFAULT_HTYPE, meta_overwrite: dict={}) -> Meta:
    required_meta = _required_meta_from_htype(htype)

    # TODO: validate overwrite meta

    required_meta.update(meta_overwrite)

    return Meta(key, storage, required_meta)

def load_tensor_meta(key: str, storage: StorageProvider) -> Meta:
    return Meta(key, storage)
"""----"""


class TensorMeta:
    # htype
    # chunk_size
    # dtype
    # custom_meta (dict)

    def __init__(self, htype: str, chunk_size: int=None, dtype: str=None, custom_meta: dict=None):
        # TODO: `htype` determines defaults
        # TODO: all other values overwrite `htype` defaults



        pass


def write_tensor_meta(key: str, storage: StorageProvider, meta: dict):
    storage[get_tensor_meta_key(key)] = pickle.dumps(meta)


def read_tensor_meta(key: str, storage: StorageProvider) -> dict:
    return pickle.loads(storage[get_tensor_meta_key(key)])


def default_tensor_meta(
    htype: Optional[str] = None,
    chunk_size: Optional[int] = None,
    dtype: Optional[str] = None,
    extra_meta: Optional[dict] = None,
):
    if chunk_size is None:
        chunk_size = DEFAULT_CHUNK_SIZE
    if dtype is None:
        dtype = DEFAULT_DTYPE
    if extra_meta is None:
        extra_meta = {}

    tensor_meta = extra_meta
    tensor_meta["chunk_size"] = chunk_size
    tensor_meta["dtype"] = dtype
    tensor_meta["length"] = 0
    if htype is not None:
        tensor_meta["htype"] = htype  # TODO: identify presets

    return tensor_meta


def update_tensor_meta_with_array(
    tensor_meta: dict, array: np.ndarray, batched=False
) -> dict:
    shape = array.shape
    if batched:
        shape = shape[1:]
    tensor_meta["dtype"] = str(array.dtype)
    tensor_meta["min_shape"] = shape
    tensor_meta["max_shape"] = shape

    return tensor_meta


def validate_tensor_meta(meta: dict):
    _raise_if_no_key("chunk_size", meta)
    _raise_if_condition(
        "chunk_size",
        meta,
        lambda chunk_size: chunk_size <= 0,
        "Chunk size must be greater than 0.",
    )

    _raise_if_no_key("dtype", meta)
    dtype_type = type(meta["dtype"])
    if dtype_type == str:
        _raise_if_condition(
            "dtype",
            meta,
            lambda dtype: not _is_dtype_supported_by_numpy(dtype),
            "Datatype must be supported by numpy. List of available numpy dtypes found here: https://numpy.org/doc/stable/user/basics.types.html",
        )
    else:
        _raise_if_condition(
            "dtype",
            meta,
            lambda dtype: type(dtype) != np.dtype,
            "Datatype must be of type string or numpy.dtype.",
        )


def _raise_if_no_key(key: str, meta: dict):
    if key not in meta:
        raise TensorMetaMissingKey(key, meta)


def _raise_if_condition(
    key: str, meta: dict, condition: Callable[[Any], bool], explanation: str = ""
):
    v = meta[key]
    if condition(v):
        raise TensorMetaInvalidValue(key, v, explanation)


def _is_dtype_supported_by_numpy(dtype: str) -> bool:
    try:
        np.dtype(dtype)
        return True
    except:
        return False
