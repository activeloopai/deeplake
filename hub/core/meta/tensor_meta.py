import hub
from hub.core.fast_forwarding import ffw_tensor_meta
from typing import Any, Callable, Dict, List, Tuple
import numpy as np
from hub.util.exceptions import (
    TensorMetaInvalidHtype,
    TensorMetaInvalidHtypeOverwriteValue,
    TensorMetaInvalidHtypeOverwriteKey,
    TensorMetaMissingRequiredValue,
    TensorMetaMutuallyExclusiveKeysError,
    UnsupportedCompressionError,
    TensorInvalidSampleShapeError,
)
from hub.util.json import validate_json_schema
from hub.constants import (
    REQUIRE_USER_SPECIFICATION,
    UNSPECIFIED,
)
from hub.compression import (
    COMPRESSION_ALIASES,
    get_compression_type,
    AUDIO_COMPRESSION,
    BYTE_COMPRESSION,
    IMAGE_COMPRESSION,
    VIDEO_COMPRESSION,
)
from hub.htype import (
    HTYPE_CONFIGURATIONS,
)
from hub.htype import HTYPE_CONFIGURATIONS, REQUIRE_USER_SPECIFICATION, UNSPECIFIED
from hub.core.meta.meta import Meta


class TensorMeta(Meta):
    htype: str
    dtype: str
    min_shape: List[int]
    max_shape: List[int]
    length: int
    sample_compression: str
    chunk_compression: str
    max_chunk_size: int

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
            **kwargs: Any key that the provided `htype` has can be overridden via **kwargs. For more information, check out `hub.htype`.
        """

        if htype != UNSPECIFIED:
            _validate_htype_exists(htype)
            _validate_htype_overwrites(htype, kwargs)
            _replace_unspecified_values(htype, kwargs)
            _validate_required_htype_overwrites(htype, kwargs)
            _format_values(htype, kwargs)

            required_meta = _required_meta_from_htype(htype)
            required_meta.update(kwargs)

            self._required_meta_keys = tuple(required_meta.keys())
            self.__dict__.update(required_meta)
        else:
            self._required_meta_keys = tuple()

        super().__init__()

    def set_dtype(self, dtype: np.dtype):
        """Should only be called once."""
        ffw_tensor_meta(self)

        if self.dtype is not None:
            raise ValueError(
                f"Tensor meta already has a dtype ({self.dtype}). Incoming: {dtype.name}."
            )

        if self.length > 0:
            raise ValueError("Dtype was None, but length was > 0.")

        self.dtype = dtype.name

    def update_shape_interval(self, shape: Tuple[int, ...]):
        ffw_tensor_meta(self)

        if not self.min_shape:  # both min_shape and max_shape are set together
            self.min_shape = list(shape)
            self.max_shape = list(shape)
        else:
            expected_dims = len(self.min_shape)

            if len(shape) != expected_dims:
                raise TensorInvalidSampleShapeError(shape, len(self.min_shape))

            for i, dim in enumerate(shape):
                self.min_shape[i] = min(dim, self.min_shape[i])
                self.max_shape[i] = max(dim, self.max_shape[i])

    def __getstate__(self) -> Dict[str, Any]:
        d = super().__getstate__()

        for key in self._required_meta_keys:
            d[key] = getattr(self, key)

        return d

    def __setstate__(self, state: Dict[str, Any]):
        if "chunk_compression" not in state:
            state["chunk_compression"] = None  # Backward compatibility
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

    if (
        htype == "image"
        and htype_overwrite["chunk_compression"] == UNSPECIFIED
        and htype_overwrite["sample_compression"] == UNSPECIFIED
    ):
        raise TensorMetaMissingRequiredValue(
            htype, ["chunk_compression", "sample_compression"]  # type: ignore
        )

    if htype in ("json", "list", "text"):
        compr = htype_overwrite["chunk_compression"]
        if compr in (None, UNSPECIFIED):
            compr = htype_overwrite["sample_compression"]
        if compr not in (None, UNSPECIFIED):
            if get_compression_type(compr) != BYTE_COMPRESSION:
                raise UnsupportedCompressionError(compr, htype)
    elif htype == "audio":
        if htype_overwrite["chunk_compression"] not in [UNSPECIFIED, None]:
            raise UnsupportedCompressionError("Chunk compression", htype=htype)
        elif htype_overwrite["sample_compression"] == UNSPECIFIED:
            raise TensorMetaMissingRequiredValue(
                htype, "sample_compression"  # type: ignore
            )
        elif get_compression_type(htype_overwrite["sample_compression"]) not in (
            None,
            AUDIO_COMPRESSION,
        ):
            raise UnsupportedCompressionError(
                htype_overwrite["sample_compression"], htype="audio"
            )

    if htype == "video":
        if htype_overwrite["chunk_compression"] not in [UNSPECIFIED, None]:
            raise UnsupportedCompressionError("Chunk compression", htype=htype)
        elif htype_overwrite["sample_compression"] == UNSPECIFIED:
            raise TensorMetaMissingRequiredValue(
                htype, "sample compression"  # type: ignore
            )
        elif get_compression_type(htype_overwrite["sample_compression"]) not in (
            None,
            VIDEO_COMPRESSION,
        ):
            raise UnsupportedCompressionError(
                htype_overwrite["sample_compression"], htype="video"
            )


def _replace_unspecified_values(htype: str, htype_overwrite: dict):
    """Replaces `UNSPECIFIED` values in `htype_overwrite` with the `htype`'s defaults."""

    defaults = HTYPE_CONFIGURATIONS[htype]

    for k, v in htype_overwrite.items():
        if v == UNSPECIFIED:
            htype_overwrite[k] = defaults[k]

    if htype in ("json", "list", "text") and not htype_overwrite["dtype"]:
        htype_overwrite["dtype"] = HTYPE_CONFIGURATIONS[htype]["dtype"]


def _validate_required_htype_overwrites(htype: str, htype_overwrite: dict):
    """Raises errors if `htype_overwrite` has invalid values."""

    sample_compression = htype_overwrite["sample_compression"]
    sample_compression = COMPRESSION_ALIASES.get(sample_compression, sample_compression)
    if sample_compression not in hub.compressions:
        raise UnsupportedCompressionError(sample_compression)

    chunk_compression = htype_overwrite["chunk_compression"]
    chunk_compression = COMPRESSION_ALIASES.get(chunk_compression, chunk_compression)
    if chunk_compression not in hub.compressions:
        raise UnsupportedCompressionError(chunk_compression)

    if sample_compression and chunk_compression:
        raise TensorMetaMutuallyExclusiveKeysError(
            custom_message="Specifying both sample-wise and chunk-wise compressions for the same tensor is not yet supported."
        )

    if htype_overwrite["dtype"] is not None:
        if htype in ("json", "list"):
            validate_json_schema(htype_overwrite["dtype"])
        else:
            _raise_if_condition(
                "dtype",
                htype_overwrite,
                lambda dtype: not _is_dtype_supported_by_numpy(dtype),
                "Datatype must be supported by numpy. Can be an `str`, `np.dtype`, or normal python type (like `bool`, `float`, `int`, etc.). List of available numpy dtypes found here: https://numpy.org/doc/stable/user/basics.types.html",
            )

    if htype == "text":
        if htype_overwrite["dtype"] not in (str, "str"):
            raise TensorMetaInvalidHtypeOverwriteValue(
                "dtype",
                htype_overwrite["dtype"],
                "dtype for tensors with text htype should always be `str`",
            )


def _format_values(htype: str, htype_overwrite: dict):
    """Replaces values in `htype_overwrite` with consistent types/formats."""

    dtype = htype_overwrite["dtype"]
    if dtype is not None:
        if htype in ("json", "list"):
            if getattr(dtype, "__module__", None) == "typing":
                htype_overwrite["dtype"] = str(dtype)
        else:
            htype_overwrite["dtype"] = np.dtype(htype_overwrite["dtype"]).name

    for key, value in COMPRESSION_ALIASES.items():
        if htype_overwrite.get("sample_compression") == key:
            htype_overwrite["sample_compression"] = value
        if htype_overwrite.get("chunk_compression") == key:
            htype_overwrite["chunk_compression"] = value


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
