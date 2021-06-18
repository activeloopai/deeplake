from hub.util.compress import compress_array
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
        compression: str = UNCOMPRESSED,
    ):
        if (path is None) == (array is None):
            raise ValueError("Must pass either `path` or `array`.")

        if path is not None:
            self.path = pathlib.Path(path)
            self._array = None
            if compression != UNCOMPRESSED:
                # TODO: maybe this should be possible? this may help make code more concise
                raise ValueError(
                    "Should not pass `compression` with a `path`."
                )  # TODO: better message

        if array is not None:
            self.path = None
            self._array = array

            # TODO: validate compression?
            self._compression = compression

    @property
    def is_symbolic(self) -> bool:
        return self._array is None

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
    def compression(self) -> str:
        self.read()
        return self._compression.lower()

    def compressed_bytes(self) -> bytes:
        """If `self` represents a compressed sample, this will return the raw compressed bytes."""

        # TODO: compressed bytes should be stripped of all meta -- this meta should be relocated to `IndexMeta`

        if self.compression == UNCOMPRESSED:
            # TODO: test this gets raised
            raise SampleIsNotCompressedError(str(self))

        if self.path is None:
            return compress_array(self.array, self.compression)

        with open(self.path, "rb") as f:
            return f.read()

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
                self._compression = img.format
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
