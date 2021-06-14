from typing import Any, Callable, List, Tuple
import numpy as np
from hub.util.exceptions import (
    TensorMetaInvalidHtype,
    TensorMetaInvalidHtypeOverwriteValue,
    TensorMetaInvalidHtypeOverwriteKey,
)
from hub.util.callbacks import CallbackList
from hub.util.keys import get_tensor_meta_key
from hub.constants import DEFAULT_CHUNK_SIZE
from hub.htypes import DEFAULT_HTYPE, HTYPE_CONFIGURATIONS
from hub.core.storage.provider import StorageProvider
from hub.core.meta.meta import Meta


def _remove_none_values_from_dict(d: dict) -> dict:
    new_d = {}
    for k, v in d.items():
        if v is not None:
            new_d[k] = v
    return new_d


class TensorMeta(Meta):
    htype: str
    dtype: str
    min_shape: List[int]
    max_shape: List[int]
    chunk_size: int
    length: int

    @staticmethod
    def create(
        key: str,
        storage: StorageProvider,
        htype: str = DEFAULT_HTYPE,
        **kwargs,
    ):
        htype_overwrite = _remove_none_values_from_dict(dict(kwargs))
        _validate_htype_overwrites(htype, htype_overwrite)

        required_meta = _required_meta_from_htype(htype)
        required_meta.update(htype_overwrite)

        return TensorMeta(
            get_tensor_meta_key(key), storage, required_meta=required_meta
        )

    @staticmethod
    def load(key: str, storage: StorageProvider):
        return TensorMeta(get_tensor_meta_key(key), storage)

    def update_tensor_meta_with_array(self, array: np.ndarray, batched=False):
        shape = array.shape
        if batched:
            shape = shape[1:]

        self.dtype = str(array.dtype)
        self.min_shape = list(shape)
        self.max_shape = list(shape)

    def update_shape_interval(self, shape: Tuple[int]):
        for i, dim in enumerate(shape):
            self.min_shape[i] = min(dim, self.min_shape[i])
            self.max_shape[i] = max(dim, self.max_shape[i])


def _required_meta_from_htype(htype: str) -> dict:
    _check_valid_htype(htype)
    defaults = HTYPE_CONFIGURATIONS[htype]

    required_meta = {
        "htype": htype,
        "dtype": defaults["dtype"],
        "chunk_size": DEFAULT_CHUNK_SIZE,
        "min_shape": CallbackList,
        "max_shape": CallbackList,
        "length": 0,
    }

    required_meta.update(defaults)
    return required_meta


def _validate_htype_overwrites(htype: str, htype_overwrite: dict):
    _check_valid_htype(htype)
    defaults = HTYPE_CONFIGURATIONS[htype]

    for key in htype_overwrite.keys():
        if key not in defaults:
            raise TensorMetaInvalidHtypeOverwriteKey(htype, key, list(defaults.keys()))

    if "chunk_size" in htype_overwrite:
        _raise_if_condition(
            "chunk_size",
            htype_overwrite,
            lambda chunk_size: chunk_size <= 0,
            "Chunk size must be greater than 0.",
        )

    if "dtype" in htype_overwrite:
        if type(htype_overwrite["dtype"]) != str:
            # TODO: support np.dtype alongside str
            raise TensorMetaInvalidHtypeOverwriteValue(
                "dtype", htype_overwrite["dtype"], "dtype must be of type `str`."
            )

        _raise_if_condition(
            "dtype",
            htype_overwrite,
            lambda dtype: not _is_dtype_supported_by_numpy(dtype),
            "Datatype must be supported by numpy. List of available numpy dtypes found here: https://numpy.org/doc/stable/user/basics.types.html",
        )


def _check_valid_htype(htype: str):
    if htype not in HTYPE_CONFIGURATIONS:
        raise TensorMetaInvalidHtype(htype, list(HTYPE_CONFIGURATIONS.keys()))


def _raise_if_condition(
    key: str, meta: dict, condition: Callable[[Any], bool], explanation: str = ""
):
    v = meta[key]
    if condition(v):
        raise TensorMetaInvalidHtypeOverwriteValue(key, v, explanation)


def _is_dtype_supported_by_numpy(dtype: str) -> bool:
    try:
        np.dtype(dtype)
        return True
    except:
        return False
