from hub.core.compression import compress_array
from hub.constants import UNCOMPRESSED
from hub.core.chunk_engine.flatten import row_wise_to_bytes
import numpy as np
import pathlib
from typing import List, Optional, Tuple
from hub.util.exceptions import (
    HubAutoUnsupportedFileExtensionError,
    SampleCorruptedError,
    SampleIsNotCompressedError,
)

from PIL import Image  # type: ignore


IMAGE_SUFFIXES: List[str] = [".jpeg", ".jpg", ".png"]
SUPPORTED_SUFFIXES: List[str] = IMAGE_SUFFIXES


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

    @property
    def is_symbolic(self) -> bool:
        return self._array is None

    @property
    def is_empty(self) -> bool:
        self.read()
        return 0 in self.array.shape

    @property
    def array(self) -> np.ndarray:
        self.read()
        return self._array  # type: ignore

    @property
    def shape(self) -> Tuple[int, ...]:
        self.read()
        return self._array.shape  # type: ignore

    @property
    def dtype(self) -> str:
        self.read()
        return self._array.dtype.name  # type: ignore

    @property
    def original_compression(self) -> str:
        self.read()

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
        if self.path is not None and self.original_compression == compression:
            with open(self.path, "rb") as f:
                return f.read()

        return compress_array(self.array, compression)

    def uncompressed_bytes(self) -> bytes:
        """Returns `self.array` as uncompressed bytes."""

        # TODO: get flatten function (row_wise_to_bytes) from tensor_meta
        return row_wise_to_bytes(self.array)

    def read(self):
        if self._array is None:
            # TODO: explain this
            if self._array is not None:
                return self._array

            suffix = self.path.suffix.lower()

            if suffix in IMAGE_SUFFIXES:
                img = Image.open(self.path)

                # TODO: mention in docstring that if this loads correctly, the meta is assumed to be valid
                try:
                    self._array = np.array(img)
                except:
                    # TODO: elaborate on why it corrupted
                    raise SampleCorruptedError(self.path)

                # TODO: validate compression?
                self._original_compression = img.format.lower()
                return self._array

            raise HubAutoUnsupportedFileExtensionError(self._suffix, SUPPORTED_SUFFIXES)

    def __str__(self):
        if self.is_symbolic:
            return f"Sample(is_symbolic=True, path={self.path})"

        return f"Sample(is_symbolic=False, shape={self.shape}, compression='{self.compression}', dtype='{self.dtype}' path={self.path})"

    def __repr__(self):
        return str(self)


def load(path: str) -> Sample:
    """Utility that loads data from a file into a `np.ndarray` in 1 line of code. Also provides access to all important metadata.

    Note:
        No data is actually loaded until you try to read a property of the returned `Sample`. This is useful for passing along to
            `tensor.append` and `tensor.extend`.

    Examples:
        >>> sample = hub.load("path/to/cat.jpeg")
        >>> type(sample.array)
        <class 'numpy.ndarray'>
        >>> sample.compression
        'jpeg'

    Supported File Types:
        image: png, jpeg, and all others supported by `PIL`: https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#fully-supported-formats

    Args:
        path (str): Path to a supported file.

    Returns:
        Sample: Sample object. Call `sample.array` to get the `np.ndarray`.
    """

    return Sample(path)
