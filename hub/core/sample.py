# type: ignore
from hub.core.compression import (
    compress_array,
    verify_compressed_file,
    read_meta_from_compressed_file,
    get_compression,
)
from hub.util.exceptions import CorruptedSampleError
import numpy as np
from typing import List, Optional, Tuple, Union

from PIL import Image  # type: ignore
from io import BytesIO


class Sample:
    path: Optional[str]

    def __init__(
        self,
        path: str = None,
        array: np.ndarray = None,
        verify: bool = False,
    ):
        """Represents a single sample for a tensor. Provides all important meta information in one place.

        Note:
            If `self.is_lazy` is True, this `Sample` doesn't actually have any data loaded. To read this data,
                simply try to read it into a numpy array (`sample.array`)

        Args:
            path (str): Path to a sample stored on the local file system that represents a single sample. If `path` is provided, `array` should not be.
                Implicitly makes `self.is_lazy == True`.
            array (np.ndarray): Array that represents a single sample. If `array` is provided, `path` should not be. Implicitly makes `self.is_lazy == False`.
            verify (bool): If a path is provided, verifies the sample if True.

        Raises:
            ValueError: Cannot create a sample from both a `path` and `array`.
        """

        if (path is None) == (array is None):
            raise ValueError("Must pass either `path` or `array`.")

        self._compressed_bytes = {}
        self._uncompressed_bytes = None

        if path is not None:
            self.path = path
            self._array = None
            self._typestr = None
            self._shape = None
            self._compression = None
            self._verified = False
            self._verify = verify

        if array is not None:
            self.path = None
            self._array = array
            self._shape = array.shape
            self._typestr = array.__array_interface__["typestr"]
            self._compression = None

    @property
    def dtype(self):
        self._read_meta()
        return np.dtype(self._typestr).name

    @property
    def shape(self):
        self._read_meta()
        return self._shape

    @property
    def compression(self):
        self._read_meta()
        return self._compression

    def _read_meta(self, f=None):
        if self._shape is not None:
            return
        if f is None:
            f = self.path
        self._compression, self._shape, self._typestr = read_meta_from_compressed_file(
            f
        )

    @property
    def is_lazy(self) -> bool:
        return self._array is None

    @property
    def is_empty(self) -> bool:
        return 0 in self.shape

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

        compressed_bytes = self._compressed_bytes.get(compression)
        if compressed_bytes is None:
            if self.path is not None:
                with open(self.path, "rb") as f:
                    compressed_bytes = f.read()
                self._compression = get_compression(compressed_bytes[:32])
                if self._compression == compression:
                    if self._verify:
                        self._shape, self._typestr = verify_compressed_file(
                            compressed_bytes, self._compression
                        )
                    else:
                        _, self._shape, self._typestr = read_meta_from_compressed_file(
                            compressed_bytes, compression=self._compression
                        )
                else:
                    img = Image.open(BytesIO(compressed_bytes))
                    if img.mode == "1":
                        self._uncompressed_bytes = img.tobytes("raw", "L")
                    else:
                        self._uncompressed_bytes = img.tobytes()
                    compressed_bytes = compress_array(self.array, compression)
            else:
                compressed_bytes = compress_array(self.array, compression)
            self._compressed_bytes[compression] = compressed_bytes
        return compressed_bytes

    def uncompressed_bytes(self) -> bytes:
        """Returns uncompressed bytes."""

        if self._uncompressed_bytes is None:
            if self.path is not None:
                img = Image.open(self.path)
                if img.mode == "1":
                    # Binary images need to be extended from bits to bytes
                    self._uncompressed_bytes = img.tobytes("raw", "L")
                else:
                    self._uncompressed_bytes = img.tobytes()
            else:
                self._uncompressed_bytes = self._array.tobytes()

        return self._uncompressed_bytes

    @property
    def array(self) -> np.ndarray:

        if self._array is None:
            self._read_meta()
            array_interface = {
                "shape": self._shape,
                "typestr": self._typestr,
                "version": 3,
                "data": self.uncompressed_bytes(),
            }

            class ArrayData:
                __array_interface__ = array_interface

            self._array = np.array(ArrayData, None)
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
