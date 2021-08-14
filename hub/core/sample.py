# type: ignore
from hub.core.compression import compress_array
import numpy as np
import pathlib
from typing import List, Optional, Tuple, Union

from PIL import Image  # type: ignore


class Sample:
    path: Optional[pathlib.Path]

    def __init__(
        self,
        path: str = None,
        array: np.ndarray = None,
    ):
        """Represents a single sample for a tensor. Provides all important meta information in one place.

        Note:
            If `self.is_lazy` is True, this `Sample` doesn't actually have have any data loaded. To read this data,
                simply try to read it into a numpy array (`sample.array`)

        Args:
            path (str): Path to a sample stored on the local file system that represents a single sample. If `path` is provided, `array` should not be.
                Implicitly makes `self.is_lazy == True`.
            array (np.ndarray): Array that represents a single sample. If `array` is provided, `path` should not be. Implicitly makes `self.is_lazy == False`.

        Raises:
            ValueError: Cannot create a sample from both a `path` and `array`.
        """

        if (path is None) == (array is None):
            raise ValueError("Must pass either `path` or `array`.")

        if path is not None:
            self.path = pathlib.Path(path)
            self._array = None
            self._read_meta()

        if array is not None:
            self.path = None
            self._array = array
            self.shape = array.shape
            self.dtype = array.dtype.name
            self.compression = None

        self._compressed_bytes = None
        self._uncompressed_bytes = None

    @property
    def is_lazy(self) -> bool:
        return self._array is None

    @property
    def is_empty(self) -> bool:
        return 0 in self._shape

    @property
    def array(self) -> np.ndarray:
        self._read()
        return self._array  # type: ignore

    def compressed_bytes(self, compression: str) -> bytes:
        """Returns this sample as compressed bytes.

        Note:
            If this sample is pointing to a path and the requested `compression` is the same as it's stored in, the data is
                returned without re-compressing.

        Args:
            compression (str): `self.array` will be compressed into this format. If `compression is None`, return `self.uncompressed_bytes()`.

        Returns:
            bytes: Bytes for the compressed sample. Contains all metadata required to decompress within these bytes.
        """

        if compression is None:
            return self.uncompressed_bytes()

        if self._compressed_bytes is None:

            # if the sample is already compressed in the requested format, just return the raw bytes
            if self.path is not None and self.compression == compression:

                with open(self.path, "rb") as f:
                    self._compressed_bytes = f.read()

            else:
                self._compressed_bytes = compress_array(self.array, compression)

        return self._compressed_bytes

    def uncompressed_bytes(self) -> bytes:
        """Returns `self.array` as uncompressed bytes."""

        if self._uncompressed_bytes is None:
            self._uncompressed_bytes = self.array.tobytes()

        return self._uncompressed_bytes

    def _read_meta(self):
        """Reads shape, dtype and format without decompressing the sample."""
        img = Image.open(self.path)
        self.shape, dtype = Image._conv_type_shape(img)
        self.dtype = np.dtype(dtype).name
        self.compression = img.format.lower()

    def _read(self):
        """If this sample hasn't been already read into memory, do so."""
        # "cache"
        if self._array is not None:
            return self._array

        # TODO: raise a comprehensive error for unsupported compression types

        # path will definitely not be `None` because of `__init__`
        img = Image.open(self.path)
        self._array = np.array(img)
        return self._array

    def __str__(self):
        if self.is_lazy:
            return f"Sample(is_lazy=True, path={self.path})"

        return f"Sample(is_lazy=False, shape={self.shape}, compression='{self.compression}', dtype='{self.dtype}' path={self.path})"

    def __repr__(self):
        return str(self)

    def __array__(self):
        return self.array


SampleValue = Union[np.ndarray, int, float, bool, Sample]
