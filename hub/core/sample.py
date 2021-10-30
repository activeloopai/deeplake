# type: ignore
from hub.core.compression import (
    compress_array,
    compress_bytes,
    decompress_array,
    verify_compressed_file,
    read_meta_from_compressed_file,
    get_compression,
    _to_hub_mkv,
)
from hub.compression import (
    get_compression_type,
    AUDIO_COMPRESSION,
    IMAGE_COMPRESSION,
    VIDEO_COMPRESSION,
)
from hub.compression import get_compression_type, AUDIO_COMPRESSION, IMAGE_COMPRESSION
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
        buffer: Union[bytes, memoryview] = None,
        compression: str = None,
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
            buffer: (bytes): Byte buffer that represents a single sample. If compressed, `compression` argument should be provided.
            compression (str): Specify in case of byte buffer.
            verify (bool): If a path is provided, verifies the sample if True.

        Raises:
            ValueError: Cannot create a sample from both a `path` and `array`.
        """

        if not any((path, array, buffer)):
            raise ValueError("Must pass one of `path`, `array` or `buffer`.")

        self._compressed_bytes = {}
        self._uncompressed_bytes = None

        self._array = None
        self._typestr = None
        self._shape = None
        self.path = None
        self._buffer = None

        if path is not None:
            self.path = path
            self._compression = None
            self._verified = False
            self._verify = verify

        if array is not None:
            self._array = array
            self._shape = array.shape
            self._typestr = array.__array_interface__["typestr"]
            self._compression = None

        if buffer is not None:
            self._compression = compression
            self._buffer = buffer
            if compression is None:
                self._uncompressed_bytes = buffer
            else:
                self._compressed_bytes[compression] = buffer

    @property
    def buffer(self):
        if self._buffer is not None:
            return self._buffer
        return self.compressed_bytes(self.compression)

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
        if self._compression is None and self.path:
            self._read_meta()
        return self._compression

    def _read_meta(self, f=None):
        if self._shape is not None:
            return
        if f is None:
            f = self.path if self.path else self.compressed_bytes[self._compression]
        self._compression, self._shape, self._typestr = read_meta_from_compressed_file(
            f
        )

    @property
    def is_lazy(self) -> bool:
        return self._array is None

    @property
    def is_empty(self) -> bool:
        return 0 in self.shape

    def _recompress(self, buffer: bytes, compression: str) -> bytes:
        if get_compression_type(self._compression) != IMAGE_COMPRESSION:
            raise ValueError(
                "Recompression with different format is only supported for images."
            )
        img = Image.open(BytesIO(buffer))
        if img.mode == "1":
            self._uncompressed_bytes = img.tobytes("raw", "L")
        else:
            self._uncompressed_bytes = img.tobytes()
        return compress_array(self.array, compression)

    def compressed_bytes(self, compression: str) -> bytes:
        """Returns this sample as compressed bytes.

        Note:
            If this sample is pointing to a path and the requested `compression` is the same as it's stored in, the data is
                returned without re-compressing.

        Args:
            compression (str): `self.array` will be compressed into this format. If `compression is None`, return `self.uncompressed_bytes()`.

        Returns:
            bytes: Bytes for the compressed sample. Contains all metadata required to decompress within these bytes.

        Raises:
            ValueError: On recompression of unsupported formats.
        """

        if compression is None:
            return self.uncompressed_bytes()

        compressed_bytes = self._compressed_bytes.get(compression)
        if compressed_bytes is None:
            if self.path is not None:
                if self._compression is None:
                    self._compression = get_compression(path=self.path)
                if self._compression in (
                    "mp4",
                    "mkv",
                ):  # mp4 byte stream is not seekable, may not be able to extract duration from mkv byte stream
                    compressed_bytes = _to_hub_mkv(self.path)
                else:
                    with open(self.path, "rb") as f:
                        compressed_bytes = f.read()
                    if self._compression is None:
                        self._compression = get_compression(
                            header=compressed_bytes[:32]
                        )
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
                    compressed_bytes = self._recompress(compressed_bytes, compression)
            elif self._buffer is not None:
                compressed_bytes = self._recompress(self._buffer, compression)
            else:
                compressed_bytes = compress_array(self.array, compression)
            self._compressed_bytes[compression] = compressed_bytes
        return compressed_bytes

    def uncompressed_bytes(self) -> bytes:
        """Returns uncompressed bytes."""

        if self._uncompressed_bytes is None:
            if self.path is not None:
                compr = self._compression
                if compr is None:
                    compr = get_compression(path=self.path)
                if get_compression_type(compr) in (
                    AUDIO_COMPRESSION,
                    VIDEO_COMPRESSION,
                ):
                    self._compression = compr
                    if self._array is None:
                        self._array = decompress_array(self.path, compression=compr)
                    self._uncompressed_bytes = self._array.tobytes()
                else:
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
            compr = self._compression
            if compr is None:
                compr = get_compression(path=self.path)
            if get_compression_type(compr) in (AUDIO_COMPRESSION, VIDEO_COMPRESSION):
                self._compression = compr
                array = decompress_array(self.path or self._buffer, compression=compr)
                if self._shape is None:
                    self._shape = array.shape
                    self._typestr = array.__array_interface__["typestr"]
                self._array = array
            else:
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

    def __eq__(self, other):
        if self.path is not None and other.path is not None:
            return self.path == other.path
        return self.buffer == other.buffer


SampleValue = Union[np.ndarray, int, float, bool, Sample]
