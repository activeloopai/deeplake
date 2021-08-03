from typing import Any, Callable, Dict, List, Tuple
import numpy as np
from hub.util.exceptions import (
    TensorInvalidSampleShapeError,
    TensorMetaInvalidHtype,
    TensorMetaInvalidHtypeOverwriteValue,
    TensorMetaInvalidHtypeOverwriteKey,
    TensorDtypeMismatchError,
    TensorMetaMissingRequiredValue,
    UnsupportedCompressionError,
)
from hub.constants import (
    REQUIRE_USER_SPECIFICATION,
    SUPPORTED_COMPRESSIONS,
    COMPRESSION_ALIASES,
    UNSPECIFIED,
)
from hub.htypes import HTYPE_CONFIGURATIONS
from hub.core.meta.meta import Meta


class TensorMeta(Meta):
    htype: str
    dtype: str
    min_shape: List[int]
    max_shape: List[int]
    length: int
    sample_compression: str

    def __init__(
        self,
        htype: str = UNSPECIFIED,
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

        if htype != UNSPECIFIED:
            _validate_htype_exists(htype)
            _validate_htype_overwrites(htype, kwargs)
            _replace_unspecified_values(htype, kwargs)
            _validate_required_htype_overwrites(kwargs)
            _format_values(kwargs)

            required_meta = _required_meta_from_htype(htype)
            required_meta.update(kwargs)

            self._required_meta_keys = tuple(required_meta.keys())
            self.__dict__.update(required_meta)
        else:
            self._required_meta_keys = tuple()

        super().__init__()

    def update(self, shape: Tuple[int], dtype, num_samples: int):
        """Update `self.min_shape` and `self.max_shape`, `dtype` (if it is None), and increment length with `num_samples`.

        Args:
            shape (Tuple[int]): [description]
            dtype ([type]): [description]
            num_samples (int): [description]

        Raises:
            ValueError: [description]
        """

        # TODO: remove this entire function
        raise NotImplementedError

        if num_samples < 0:
            raise ValueError(
                f"Num samples cannot be negative. Got: '{num_samples}'"
            )

        dtype = np.dtype(dtype)

        if self.length <= 0:
            if self.dtype is None:
                self.dtype = dtype.name

            self.min_shape = list(shape)
            self.max_shape = list(shape)
        else:
            # update meta subsequent times
            self._update_shape_interval(shape)

        self.length += num_samples

    def set_dtype(self, dtype: np.dtype):
        # TODO: docstring

        if self.dtype is not None:
            raise ValueError(f"Tensor meta already has a dtype ({self.dtype}). Incoming: {dtype.name}.")

        if self.length > 0:
            raise ValueError("Dtype was None, but length was > 0.")

        self.dtype = dtype.name

    def update_shape_interval(self, shape: Tuple[int, ...]):
        if self.length <= 0:
            self.min_shape = list(shape)
            self.max_shape = list(shape)
        for i, dim in enumerate(shape):
            self.min_shape[i] = min(dim, self.min_shape[i])
            self.max_shape[i] = max(dim, self.max_shape[i])

    def __getstate__(self) -> Dict[str, Any]:
        d = super().__getstate__()

        for key in self._required_meta_keys:
            d[key] = getattr(self, key)

        return d

    def __setstate__(self, state: Dict[str, Any]):
        super().__setstate__(state)
        self._required_meta_keys = tuple(state.keys())

    @property
    def nbytes(self):
        # TODO: optimize this
        return len(self.tobytes())

    def __str__(self):
        return str(self.__getstate__())


def _required_meta_from_htype(htype: str) -> dict:
    """Gets a dictionary with all required meta information to define a tensor."""

    _validate_htype_exists(htype)
    defaults = HTYPE_CONFIGURATIONS[htype]

    required_meta = {
        "htype": htype,
        "min_shape": [],
        "max_shape": [],
        "length": 0,
        **defaults,
    }

    return required_meta


def _validate_htype_overwrites(htype: str, htype_overwrite: dict):
    """Raises errors if `htype_overwrite` has invalid keys or was missing required values."""

    defaults = HTYPE_CONFIGURATIONS[htype]

    for key, value in htype_overwrite.items():
        if key not in defaults:
            raise TensorMetaInvalidHtypeOverwriteKey(htype, key, list(defaults.keys()))

        if value == UNSPECIFIED:
            if defaults[key] == REQUIRE_USER_SPECIFICATION:
                raise TensorMetaMissingRequiredValue(htype, key)


def _replace_unspecified_values(htype: str, htype_overwrite: dict):
    """Replaces `UNSPECIFIED` values in `htype_overwrite` with the `htype`'s defaults."""

    defaults = HTYPE_CONFIGURATIONS[htype]

    for k, v in htype_overwrite.items():
        if v == UNSPECIFIED:
            htype_overwrite[k] = defaults[k]


def _validate_required_htype_overwrites(htype_overwrite: dict):
    """Raises errors if `htype_overwrite` has invalid values."""

    if htype_overwrite["sample_compression"] not in SUPPORTED_COMPRESSIONS:
        raise UnsupportedCompressionError(htype_overwrite["sample_compression"])

    if htype_overwrite["dtype"] is not None:
        _raise_if_condition(
            "dtype",
            htype_overwrite,
            lambda dtype: not _is_dtype_supported_by_numpy(dtype),
            "Datatype must be supported by numpy. Can be an `str`, `np.dtype`, or normal python type (like `bool`, `float`, `int`, etc.). List of available numpy dtypes found here: https://numpy.org/doc/stable/user/basics.types.html",
        )


def _format_values(htype_overwrite: dict):
    """Replaces values in `htype_overwrite` with consistent types/formats."""

    if htype_overwrite["dtype"] is not None:
        htype_overwrite["dtype"] = np.dtype(htype_overwrite["dtype"]).name

    for key, value in COMPRESSION_ALIASES.items():
        if htype_overwrite.get("sample_compression") == key:
            htype_overwrite["sample_compression"] = value


def _validate_htype_exists(htype: str):
    """Raises errors if given an unrecognized htype."""
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
