import json
from typing import Any, Callable, Optional

import numpy as np

import hub
from hub.constants import DEFAULT_CHUNK_SIZE, DEFAULT_DTYPE, META_ENCODING
from hub.core.typing import StorageProvider
from hub.util.exceptions import TensorMetaInvalidValue, TensorMetaMissingKey
from hub.util.keys import get_tensor_meta_key
from hub.util.array import normalize_and_batchify_array_shape


def write_tensor_meta(key: str, storage: StorageProvider, meta: dict):
    storage[get_tensor_meta_key(key)] = json.dumps(meta).encode(META_ENCODING)


def read_tensor_meta(key: str, storage: StorageProvider) -> dict:
    return json.loads(storage[get_tensor_meta_key(key)])


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
    tensor_meta["version"] = hub.__version__
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
    tensor_meta["min_shape"] = list(shape)
    tensor_meta["max_shape"] = list(shape)

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
