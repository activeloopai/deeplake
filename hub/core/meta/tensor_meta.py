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
    htype: str
    dtype: str
    min_shape: List[int]
    max_shape: List[int]
    chunk_size: int
    length: int
    sample_compression: str
    chunk_compression: str

    def __init__(
        self,
        htype: str = DEFAULT_HTYPE,
        **kwargs,
    ):
        """Tensor metadata is responsible for keeping track of global sample metadata within a tensor.

        Note:
            Tensor metadata that is automatically synchronized with `storage`. For more details, see the `Meta` class.
            Auto-populates `required_meta` that `Meta` accepts as an argument.

        Args:
            htype (str): All tensors require an `htype`. This determines the default meta keys/values.
            **kwargs: Any key that the provided `htype` has can be overridden via **kwargs. For more information, check out `hub.htypes`.
        """

        htype_overwrite = _remove_none_values_from_dict(dict(kwargs))
        _validate_htype_overwrites(htype, htype_overwrite)

        required_meta = _required_meta_from_htype(htype)
        required_meta.update(htype_overwrite)
        _validate_compression(required_meta)

        self.__dict__.update(required_meta)

        super().__init__()

    def check_compatibility(self, shape: Tuple[int], dtype):
        """Checks if this tensor meta is compatible with the incoming sample(s) properties.

        Args:
            shape (Tuple[int]): Shape all samples having their compatibility checked. Must be a single-sample shape
                but can represent multiple.
            dtype: Datatype for the sample(s).

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
            expected_shape_len = len(self.min_shape)
            actual_shape_len = len(shape)
            if expected_shape_len != actual_shape_len:
                raise TensorInvalidSampleShapeError(
                    "Sample shape length is expected to be {}, actual length is {}.".format(
                        expected_shape_len, actual_shape_len
                    ),
                    shape,
                )

    def update(self, shape: Tuple[int], dtype, num_samples: int):
        """Update `self.min_shape` and `self.max_shape`, `dtype` (if it is None), and increment length with `num_samples`.

        Args:
            shape (Tuple[int]): [description]
            dtype ([type]): [description]
            num_samples (int): [description]

        Raises:
            ValueError: [description]
        """

        if num_samples <= 0:
            raise ValueError(
                f"Can only update tensor meta when the number of samples is > 0. Got: '{num_samples}'"
            )

        dtype = np.dtype(dtype)

        if self.length <= 0:
            if not self.dtype:
                self.dtype = str(dtype)

            self.min_shape = list(shape)
            self.max_shape = list(shape)
        else:
            # update meta subsequent times
            self._update_shape_interval(shape)

        self.length += num_samples

    def _update_shape_interval(self, shape: Tuple[int, ...]):
        if self.length <= 0:
            self.min_shape = list(shape)
            self.max_shape = list(shape)
        for i, dim in enumerate(shape):
            self.min_shape[i] = min(dim, self.min_shape[i])
            self.max_shape[i] = max(dim, self.max_shape[i])

    def as_dict(self):
        # TODO: tensor meta as_dict
        raise NotImplementedError


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
