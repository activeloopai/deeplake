from typing import Any, Callable, List, Sequence, Tuple, Union
import numpy as np
from hub.util.exceptions import (
    TensorInvalidSampleShapeError,
    TensorMetaInvalidHtype,
    TensorMetaInvalidHtypeOverwriteValue,
    TensorMetaInvalidHtypeOverwriteKey,
    TensorDtypeMismatchError,
    UnsupportedCompressionError,
)
from hub.util.keys import get_tensor_meta_key
from hub.constants import (
    DEFAULT_HTYPE,
    SUPPORTED_COMPRESSIONS,
    UNCOMPRESSED,
)
from hub.htypes import HTYPE_CONFIGURATIONS
from hub.core.storage.provider import StorageProvider
from hub.core.meta.meta import Meta


def _remove_none_values_from_dict(d: dict) -> dict:
    new_d = {}
    for k, v in d.items():
        if v is not None:
            new_d[k] = v
    return new_d


class TensorMeta(Meta):
    _htype: str
    _dtype: str
    _min_shape: List[int]
    _max_shape: List[int]
    _chunk_size: int
    _length: int
    _sample_compression: str
    _chunk_compression: str

    def __init__(
        self,
        tensor_meta_key: str,
        storage: StorageProvider,
        htype: str = DEFAULT_HTYPE,
        **kwargs,
    ):

        kwargs = _remove_none_values_from_dict(kwargs.copy())
        _validate_htype_overwrites(htype, kwargs)
        meta_dict = _meta_dict_from_htype(htype)
        meta_dict.update(kwargs)

        for k, v in meta_dict.items():
            # we want the meta variables to have an `_` so we can protect them with properties
            setattr(self, f"_{k}", v)

        super().__init__(tensor_meta_key, storage)

    def check_compatibility(self, shape: Sequence[int], dtype):
        """Check if this `tensor_meta` is compatible with `array`. The provided `array` is treated as a single sample.

        Note:
            If no samples exist in the tensor this `tensor_meta` corresponds with, `len(array.shape)` is not checked.

        Args:
            array (np.ndarray): Array representing a sample to check compatibility with.

        Raises:
            TensorDtypeMismatchError: Dtype for array must be equal to this meta.
            TensorInvalidSampleShapeError: If a sample already exists, `len(array.shape)` has to be consistent for all arrays.
        """

        dtype = np.dtype(dtype)

        if self.dtype and self.dtype != dtype.name:
            raise TensorDtypeMismatchError(
                self.dtype,
                dtype.name,
                self.htype,
            )

        # shape length is only enforced after at least 1 sample exists.
        if self.length > 0:
            expected_shape_len = len(self._min_shape)
            actual_shape_len = len(shape)
            if expected_shape_len != actual_shape_len:
                raise TensorInvalidSampleShapeError(
                    "Sample shape length is expected to be {}, actual length is {}.".format(
                        expected_shape_len, actual_shape_len
                    ),
                    shape,
                )

    def update(self, shape: Sequence[int], dtype, num_samples: int):
        """Update this meta with the `array` properties. The provided `array` is treated as a single sample (no batch axis)!

        Note:
            If no samples exist, `min_shape` and `max_shape` are set to this array's shape.
            If samples do exist, `min_shape` and `max_shape` are updated.

        Args:
            array (np.ndarray): Unbatched array to update this meta with.
        """

        """`array` is assumed to have a batch axis."""
        self._check_readonly()

        if num_samples <= 0:
            raise ValueError(
                f"Can only update tensor meta when the number of samples is > 0. Got: '{num_samples}'"
            )

        dtype = np.dtype(dtype)

        if self.length <= 0:
            if not self.dtype:
                self._dtype = str(dtype)

            self._min_shape = list(shape)
            self._max_shape = list(shape)
        else:
            # update meta subsequent times
            self._update_shape_interval(shape, write=False)

        self.add_length(num_samples, write=False)
        self.write()

    def _update_shape_interval(self, shape: Tuple[int, ...], write: bool = True):
        self._check_readonly()
        if self.length <= 0:
            self._min_shape = list(shape)
            self._max_shape = list(shape)

        for i, dim in enumerate(shape):
            self._min_shape[i] = min(dim, self._min_shape[i])
            self._max_shape[i] = max(dim, self._max_shape[i])

        if write:
            self.write()

    @property
    def htype(self):
        return self._htype

    @property
    def length(self):
        return self._length

    def add_length(self, delta: int, write: bool = True):
        self._length += delta
        if write:
            self.write()

    @property
    def sample_compression(self):
        return self._sample_compression

    @property
    def chunk_compression(self):
        return self._chunk_compression

    @property
    def dtype(self):
        return self._dtype

    @property
    def min_shape(self):
        return tuple(self._min_shape)

    @property
    def max_shape(self):
        return tuple(self._max_shape)

    def write(self):
        super().write(
            _htype=self._htype,
            _dtype=self._dtype,
            _chunk_size=self._chunk_size,
            _min_shape=self._min_shape,
            _max_shape=self._max_shape,
            _length=self._length,
            _sample_compression=self._sample_compression,
            _chunk_compression=self._chunk_compression,
        )


def _meta_dict_from_htype(htype: str) -> dict:
    _check_valid_htype(htype)
    defaults = HTYPE_CONFIGURATIONS[htype]

    meta_dict = {
        "htype": htype,
        "dtype": defaults.get("dtype", None),
        "chunk_size": defaults["chunk_size"],
        "min_shape": [],
        "max_shape": [],
        "length": 0,
        "sample_compression": defaults["sample_compression"],
        "chunk_compression": defaults["chunk_compression"],
    }

    _validate_compression(meta_dict)

    meta_dict = _remove_none_values_from_dict(meta_dict)
    meta_dict.update(defaults)
    return meta_dict


def _validate_compression(meta_dict: dict):
    chunk_compression = meta_dict["chunk_compression"]
    if chunk_compression != UNCOMPRESSED:
        raise NotImplementedError("Chunk compression has not been implemented yet.")

    sample_compression = meta_dict["sample_compression"]
    if sample_compression not in SUPPORTED_COMPRESSIONS:
        raise UnsupportedCompressionError(sample_compression)

    if chunk_compression not in SUPPORTED_COMPRESSIONS:
        raise UnsupportedCompressionError(chunk_compression)


def _validate_htype_overwrites(htype: str, htype_overwrite: dict):
    """Raises appropriate errors if `htype_overwrite` keys/values are invalid in correspondence to `htype`. May modify `dtype` in `htype_overwrite` if it is a non-str."""

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
        _raise_if_condition(
            "dtype",
            htype_overwrite,
            lambda dtype: not _is_dtype_supported_by_numpy(dtype),
            "Datatype must be supported by numpy. Can be an `str`, `np.dtype`, or normal python type (like `bool`, `float`, `int`, etc.). List of available numpy dtypes found here: https://numpy.org/doc/stable/user/basics.types.html",
        )

        if type(htype_overwrite["dtype"]) != str:
            htype_overwrite["dtype"] = np.dtype(htype_overwrite["dtype"]).name


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
