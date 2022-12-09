from random import sample
import deeplake
from deeplake.core.fast_forwarding import ffw_tensor_meta
from typing import Any, Callable, Dict, List, Sequence, Union, Optional, Tuple
import numpy as np
from deeplake.util.exceptions import (
    TensorMetaInvalidHtype,
    TensorMetaInvalidHtypeOverwriteValue,
    TensorMetaInvalidHtypeOverwriteKey,
    TensorMetaMissingRequiredValue,
    TensorMetaMutuallyExclusiveKeysError,
    UnsupportedCompressionError,
    TensorInvalidSampleShapeError,
    InvalidTensorLinkError,
)
from deeplake.util.json import validate_json_schema
from deeplake.constants import (
    REQUIRE_USER_SPECIFICATION,
    UNSPECIFIED,
)
from deeplake.compression import (
    COMPRESSION_ALIASES,
)
from deeplake.htype import (
    HTYPE_CONFIGURATIONS,
    HTYPE_SUPPORTED_COMPRESSIONS,
    htype as HTYPE,
)
from deeplake.htype import HTYPE_CONFIGURATIONS, REQUIRE_USER_SPECIFICATION, UNSPECIFIED
from deeplake.core.meta.meta import Meta
from deeplake.core.tensor_link import get_link_transform


class TensorMeta(Meta):
    name: Optional[str] = None
    htype: str
    dtype: str
    min_shape: List[int]
    max_shape: List[int]
    length: int
    sample_compression: str
    chunk_compression: str
    max_chunk_size: int
    tiling_threshold: int
    hidden: bool
    links: Dict[str, Dict[str, Union[str, bool]]]
    is_sequence: bool
    is_link: bool
    verify: bool

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
            **kwargs: Any key that the provided `htype` has can be overridden via **kwargs. For more information, check out `deeplake.htype`.
        """

        super().__init__()
        if htype and htype not in [UNSPECIFIED, HTYPE.DEFAULT]:
            self.set_htype(htype, **kwargs)
        else:
            self.set_htype(HTYPE.DEFAULT, **kwargs)
            self.htype = None  # type: ignore

    def add_link(
        self, name, extend_f: str, update_f: Optional[str], flatten_sequence: bool
    ):
        """Link this tensor with another."""
        link = {
            "extend": extend_f,
            "flatten_sequence": flatten_sequence,
        }
        if update_f is not None:
            link["update"] = update_f
        d = {name: link}
        _validate_links(d)
        self.links.update(d)  # type: ignore
        self.is_dirty = True

    def set_hidden(self, val: bool):
        """Set visibility of tensor."""
        self.hidden = val
        self.is_dirty = True

    def set_dtype(self, dtype: np.dtype):
        """Set dtype of tensor. Should only be called once."""

        if self.dtype is not None:
            raise ValueError(
                f"Tensor meta already has a dtype ({self.dtype}). Incoming: {dtype.name}."
            )

        self.dtype = dtype.name
        self.is_dirty = True

    def set_dtype_str(self, dtype_name: str):
        self.dtype = dtype_name
        self.is_dirty = True

    def set_htype(self, htype: str, **kwargs):
        """Set htype of tensor. Should only be called once."""

        if getattr(self, "htype", None) is not None:
            raise ValueError(
                f"Tensor meta already has a htype ({self.htype}). Incoming: {htype}."
            )

        if not kwargs:
            kwargs = HTYPE_CONFIGURATIONS[htype]

        _validate_htype_exists(htype)
        _validate_htype_overwrites(htype, kwargs)
        _replace_unspecified_values(htype, kwargs)
        _validate_required_htype_overwrites(htype, kwargs)
        _format_values(htype, kwargs)

        required_meta = _required_meta_from_htype(htype)
        required_meta.update(kwargs)

        self._required_meta_keys = tuple(required_meta.keys())

        for k in self._required_meta_keys:
            if getattr(self, k, None):
                required_meta.pop(k, None)

        self.__dict__.update(required_meta)
        self.is_dirty = True
        if self.links is None:
            self.links = {}
        _validate_links(self.links)

    def update_shape_interval(self, shape: Sequence[int]):
        """Update shape interval of tensor."""
        initial_min_shape = None if self.min_shape is None else self.min_shape.copy()
        initial_max_shape = None if self.max_shape is None else self.max_shape.copy()

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

        if initial_min_shape != self.min_shape or initial_max_shape != self.max_shape:
            self.is_dirty = True

    def update_length(self, length: int):
        """Update length of tensor."""
        self.length += length
        if length != 0:
            self.is_dirty = True

    def pop(self, index):
        """Reflect popping a sample in tensor's meta."""
        self.length -= 1
        if self.length == 0:
            self.min_shape = []
            self.max_shape = []
        self.is_dirty = True

    def __getstate__(self) -> Dict[str, Any]:
        d = super().__getstate__()

        for key in self._required_meta_keys:
            d[key] = getattr(self, key)
        d["name"] = self.name

        return d

    def __setstate__(self, state: Dict[str, Any]):
        super().__setstate__(state)
        self._required_meta_keys = tuple(state.keys())
        ffw_tensor_meta(self)

    @property
    def nbytes(self):
        """Returns size of the metadata stored in bytes."""
        # TODO: optimize this
        return len(self.tobytes())

    def __str__(self):
        return str(self.__getstate__())


def _validate_links(links: dict):
    if not isinstance(links, dict):
        raise InvalidTensorLinkError()
    allowed_keys = ("extend", "update", "flatten_sequence")
    for out_tensor, args in links.items():
        if not isinstance(out_tensor, str):
            raise InvalidTensorLinkError()
        if not isinstance(args, dict):
            raise InvalidTensorLinkError()
        if "extend" not in args:
            raise InvalidTensorLinkError(
                f"extend transform not specified for link {out_tensor}"
            )
        if "flatten_sequence" not in args:
            raise InvalidTensorLinkError(
                f"flatten_sequence arg not specified for link {out_tensor}"
            )
        try:
            get_link_transform(args["extend"])
        except KeyError:
            raise InvalidTensorLinkError(f"Invalid extend transform: {args['extend']}")
        if "update" in args:
            try:
                get_link_transform(args["update"])
            except KeyError:
                raise InvalidTensorLinkError(
                    f"Invalid update transform: {args['extend']}"
                )
        for k in args:
            if k not in allowed_keys:
                raise InvalidTensorLinkError(f"Invalid key in link meta: {k}")


def _required_meta_from_htype(htype: str) -> dict:
    """Gets a dictionary with all required meta information to define a tensor."""

    _validate_htype_exists(htype)
    defaults = HTYPE_CONFIGURATIONS[htype]

    required_meta = {
        "htype": htype,
        "min_shape": [],
        "max_shape": [],
        "length": 0,
        "hidden": False,
        **defaults,
    }

    return required_meta


def _validate_htype_overwrites(htype: str, htype_overwrite: dict):
    """Raises errors if ``htype_overwrite`` has invalid keys or was missing required values."""

    defaults = HTYPE_CONFIGURATIONS[htype]

    for key, value in htype_overwrite.items():
        if key not in defaults:
            raise TensorMetaInvalidHtypeOverwriteKey(htype, key, list(defaults.keys()))

        if isinstance(value, str) and value == UNSPECIFIED:
            if defaults[key] == REQUIRE_USER_SPECIFICATION:
                raise TensorMetaMissingRequiredValue(htype, key)

    sc = htype_overwrite["sample_compression"]
    cc = htype_overwrite["chunk_compression"]
    compr = sc if cc in (None, UNSPECIFIED) else cc
    actual_htype = f"link[{htype}]" if htype_overwrite["is_link"] else htype
    if htype.startswith("image") and sc == UNSPECIFIED and cc == UNSPECIFIED:
        raise TensorMetaMissingRequiredValue(
            actual_htype, ["chunk_compression", "sample_compression"]  # type: ignore
        )
    if htype in ("audio", "video", "point_cloud", "mesh"):
        if cc not in (UNSPECIFIED, None):
            raise UnsupportedCompressionError("Chunk compression", htype=htype)
        elif sc == UNSPECIFIED:
            raise TensorMetaMissingRequiredValue(
                actual_htype, "sample_compression"  # type: ignore
            )
    supported_compressions = HTYPE_SUPPORTED_COMPRESSIONS.get(htype)
    if (
        compr
        and compr != UNSPECIFIED
        and supported_compressions
        and compr not in supported_compressions
    ):
        raise UnsupportedCompressionError(compr, htype=htype)


def _replace_unspecified_values(htype: str, htype_overwrite: dict):
    """Replaces ``UNSPECIFIED`` values in ``htype_overwrite`` with the ``htype``'s defaults."""

    defaults = HTYPE_CONFIGURATIONS[htype]

    for k, v in htype_overwrite.items():
        if isinstance(v, str) and v == UNSPECIFIED:
            htype_overwrite[k] = defaults[k]

    if (
        htype in ("json", "list", "text", "point_cloud_calibration_matrix")
        and not htype_overwrite["dtype"]
    ):
        htype_overwrite["dtype"] = HTYPE_CONFIGURATIONS[htype]["dtype"]


def _validate_required_htype_overwrites(htype: str, htype_overwrite: dict):
    """Raises errors if `htype_overwrite` has invalid values."""
    sample_compression = htype_overwrite["sample_compression"]
    sample_compression = COMPRESSION_ALIASES.get(sample_compression, sample_compression)
    if sample_compression not in deeplake.compressions:
        raise UnsupportedCompressionError(sample_compression)

    chunk_compression = htype_overwrite["chunk_compression"]
    chunk_compression = COMPRESSION_ALIASES.get(chunk_compression, chunk_compression)
    if chunk_compression not in deeplake.compressions:
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
        if htype in ("json", "list", "point_cloud_calibration_matrix"):
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
