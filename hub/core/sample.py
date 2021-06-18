from hub.core.compression import compress_array
from hub.constants import UNCOMPRESSED, USE_UNIFORM_COMPRESSION_PER_SAMPLE
from hub.core.chunk_engine.flatten import row_wise_to_bytes
import numpy as np
import pathlib
from typing import List, Optional, Tuple

from PIL import Image  # type: ignore


class Sample:
    path: Optional[pathlib.Path]

    def __init__(
        self,
        path: str = None,
        array: np.ndarray = None,
    ):
        if (path is None) == (array is None):
            raise ValueError("Must pass either `path` or `array`.")

        if path is not None:
            self.path = pathlib.Path(path)
            self._array = None

        if array is not None:
            self.path = None
            self._array = array
            self._original_compression = UNCOMPRESSED

    @property
    def is_symbolic(self) -> bool:
        return self._array is None

    @property
    def is_empty(self) -> bool:
        self._read()
        return 0 in self.array.shape

    @property
    def array(self) -> np.ndarray:
        self._read()
        return self._array  # type: ignore

    @property
    def shape(self) -> Tuple[int, ...]:
        self._read()
        return self._array.shape  # type: ignore

    @property
    def dtype(self) -> str:
        self._read()
        return self._array.dtype.name  # type: ignore

    @property
    def compression(self) -> str:
        self._read()

        if self.is_empty:
            return UNCOMPRESSED

        return self._original_compression.lower()

    def compressed_bytes(self, compression: str) -> bytes:
        """Returns this sample as compressed bytes.

        Note:
            If this sample is pointing to a path and the requested `compression` is the same as it's stored in, the data is
                returned without re-compressing.

        Args:
            compression (str): `self.array` will be compressed into this format. If `compression == UNCOMPRESSED`, return `self.uncompressed_bytes()`.
        """

        # TODO: raise a comprehensive error for unsupported compression types
        compression = compression.lower()

        if compression == UNCOMPRESSED:
            return self.uncompressed_bytes()

        # if the sample is already compressed in the requested format, just return the raw bytes
        if self.path is not None and self.compression == compression:

            with open(self.path, "rb") as f:
                return f.read()

        return compress_array(self.array, compression)

    def uncompressed_bytes(self) -> bytes:
        """Returns `self.array` as uncompressed bytes."""

        # TODO: get flatten function (row_wise_to_bytes) from tensor_meta
        return row_wise_to_bytes(self.array)

    def _read(self):
        """If this sample hasn't been already read into memory, do so. This is required for properties to be accessible."""

        # "cache"
        if self._array is not None:
            return self._array

        # TODO: raise a comprehensive error for unsupported compression types

        # path will definitely not be `None` because of `__init__`
        img = Image.open(self.path)
        self._array = np.array(img)
        self._original_compression = img.format.lower()
        return self._array

    def __str__(self):
        if self.is_symbolic:
            return f"Sample(is_symbolic=True, path={self.path})"

        return f"Sample(is_symbolic=False, shape={self.shape}, compression='{self.compression}', dtype='{self.dtype}' path={self.path})"

    def __repr__(self):
        return str(self)
