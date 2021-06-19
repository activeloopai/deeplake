from typing import Any, Callable, List, Tuple, Union
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
    htype: str
    dtype: str
    min_shape: List[int]
    max_shape: List[int]
    chunk_size: int
    length: int
    sample_compression: str
    chunk_compression: str

    @staticmethod
    def create(
        key: str,
        storage: StorageProvider,
        htype: str = DEFAULT_HTYPE,
        **kwargs,
    ):
        """Tensor metadata is responsible for keeping track of global sample metadata within a tensor.

        Note:
            Tensor metadata that is automatically synchronized with `storage`. For more details, see the `Meta` class.
            Auto-populates `required_meta` that `Meta` accepts as an argument.

        Args:
            key (str): Key relative to `storage` where this instance will be synchronized to. Will automatically add the tensor meta filename to the end.
            storage (StorageProvider): Destination of this meta.
            htype (str): All tensors require an `htype`. This determines the default meta keys/values.
            **kwargs: Any key that the provided `htype` has can be overridden via **kwargs. For more information, check out `hub.htypes`.

        Raises:
            TensorMetaInvalidHtypeOverwriteKey: If **kwargs contains unsupported keys for the provided `htype`.
            TensorMetaInvalidHtypeOverwriteValue: If **kwargs contains unsupported values for the keys of the provided `htype`.
            NotImplementedError: Chunk compression has not been implemented! # TODO: chunk compression

        Returns:
            TensorMeta: Tensor meta object.
        """

        htype_overwrite = _remove_none_values_from_dict(dict(kwargs))
        _validate_htype_overwrites(htype, htype_overwrite)

        required_meta = _required_meta_from_htype(htype)
        required_meta.update(htype_overwrite)
        _validate_compression(required_meta)

        return TensorMeta(
            get_tensor_meta_key(key), storage, required_meta=required_meta
        )

    @staticmethod
    def load(key: str, storage: StorageProvider):
        return TensorMeta(get_tensor_meta_key(key), storage)

    def check_array_sample_is_compatible(self, array: np.ndarray):
        """Check if this `tensor_meta` is compatible with `array`. The provided `array` is treated as a single sample.

        Note:
            If no samples exist in the tensor this `tensor_meta` corresponds with, `len(array.shape)` is not checked.

        Args:
            array (np.ndarray): Array representing a sample to check compatibility with.

        Raises:
            TensorDtypeMismatchError: Dtype for array must be equal to this meta.
            TensorInvalidSampleShapeError: If a sample already exists, `len(array.shape)` has to be consistent for all arrays.
        """

        if self.dtype and self.dtype != array.dtype.name:
            raise TensorDtypeMismatchError(
                self.dtype,
                array.dtype.name,
                self.htype,
            )

        # shape length is only enforced after at least 1 sample exists.
        if self.length > 0:
            expected_shape_len = len(self.min_shape)
            actual_shape_len = len(array.shape)
            if expected_shape_len != actual_shape_len:
                raise TensorInvalidSampleShapeError(
                    "Sample shape length is expected to be {}, actual length is {}.".format(
                        expected_shape_len, actual_shape_len
                    ),
                    array.shape,
                )

    def update_with_sample(self, array: np.ndarray):
        """Update this meta with the `array` properties. The provided `array` is treated as a single sample (no batch axis)!

        Note:
            If no samples exist, `min_shape` and `max_shape` are set to this array's shape.
            If samples do exist, `min_shape` and `max_shape` are updated.

        Args:
            array (np.ndarray): Unbatched array to update this meta with.
        """

        """`array` is assumed to have a batch axis."""

        shape = array.shape

        if self.length <= 0:
            if not self.dtype:
                self.dtype = str(array.dtype)

            self.min_shape = list(shape)
            self.max_shape = list(shape)
        else:
            # update meta subsequent times
            self._update_shape_interval(shape)

    def _update_shape_interval(self, shape: Tuple[int, ...]):
        if self.length <= 0:
            self.min_shape = list(shape)
            self.max_shape = list(shape)
        for i, dim in enumerate(shape):
            self.min_shape[i] = min(dim, self.min_shape[i])
            self.max_shape[i] = max(dim, self.max_shape[i])


def _required_meta_from_htype(htype: str) -> dict:
    _check_valid_htype(htype)
    defaults = HTYPE_CONFIGURATIONS[htype]

    required_meta = {
        "htype": htype,
        "dtype": defaults.get("dtype", None),
        "chunk_size": defaults["chunk_size"],
        "min_shape": [],
        "max_shape": [],
        "length": 0,
        "sample_compression": defaults["sample_compression"],
        "chunk_compression": defaults["chunk_compression"],
    }

    required_meta = _remove_none_values_from_dict(required_meta)
    required_meta.update(defaults)
    return required_meta


def _validate_compression(required_meta: dict):
    chunk_compression = required_meta["chunk_compression"]
    if chunk_compression != UNCOMPRESSED:
        raise NotImplementedError("Chunk compression has not been implemented yet.")

    sample_compression = required_meta["sample_compression"]
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
