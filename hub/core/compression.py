import hub
from hub.util.exceptions import (
    SampleCompressionError,
    SampleDecompressionError,
    UnsupportedCompressionError,
    CorruptedSampleError,
)
from hub.compression import get_compression_type
from typing import Union, Tuple, Sequence, List, Optional
import numpy as np

from PIL import Image, UnidentifiedImageError  # type: ignore
from io import BytesIO
import mmap

import lz4.frame  # type: ignore


def to_image(array: np.ndarray) -> Image:
    shape = array.shape
    if len(shape) == 3 and shape[0] != 1 and shape[2] == 1:
        # convert (X,Y,1) grayscale to (X,Y) for pillow compatibility
        return Image.fromarray(array.squeeze(axis=2))

    return Image.fromarray(array)


def compress_bytes(buffer: Union[bytes, memoryview], compression: str) -> bytes:
    if compression == "lz4":
        return lz4.frame.compress(buffer)
    else:
        raise SampleCompressionError(
            (len(buffer),), compression, f"Not a byte compression: {compression}"
        )


def decompress_bytes(buffer: Union[bytes, memoryview], compression: str) -> bytes:
    if compression == "lz4":
        return lz4.frame.decompress(buffer)
    else:
        raise SampleDecompressionError()


def compress_array(array: np.ndarray, compression: str) -> bytes:
    """Compress some numpy array using `compression`. All meta information will be contained in the returned buffer.

    Note:
        `decompress_array` may be used to decompress from the returned bytes back into the `array`.

    Args:
        array (np.ndarray): Array to be compressed.
        compression (str): `array` will be compressed with this compression into bytes. Right now only arrays compatible with `PIL` will be compressed.

    Raises:
        UnsupportedCompressionError: If `compression` is unsupported. See `hub.compressions`.
        SampleCompressionError: If there was a problem compressing `array`.

    Returns:
        bytes: Compressed `array` represented as bytes.
    """

    # empty sample shouldn't be compressed

    if 0 in array.shape:
        return bytes()

    if compression not in hub.compressions:
        raise UnsupportedCompressionError(compression)

    if compression is None:
        return array.tobytes()

    if get_compression_type(compression) == "byte":
        return compress_bytes(array.tobytes(), compression)

    try:
        img = to_image(array)
        out = BytesIO()
        out._close = out.close  # type: ignore
        out.close = (  # type: ignore
            lambda: None
        )  # sgi save handler will try to close the stream (see https://github.com/python-pillow/Pillow/pull/5645)
        kwargs = {"sizes": [img.size]} if compression == "ico" else {}
        img.save(out, compression, **kwargs)
        out.seek(0)
        compressed_bytes = out.read()
        out._close()  # type: ignore
        return compressed_bytes
    except (TypeError, OSError) as e:
        raise SampleCompressionError(array.shape, compression, str(e))


def decompress_array(
    buffer: Union[bytes, memoryview],
    shape: Optional[Tuple[int]] = None,
    dtype: Optional[str] = None,
    compression: Optional[str] = None,
) -> np.ndarray:
    """Decompress some buffer into a numpy array. It is expected that all meta information is
    stored inside `buffer`.

    Note:
        `compress_array` may be used to get the `buffer` input.

    Args:
        buffer (bytes, memoryview): Buffer to be decompressed. It is assumed all meta information required to
            decompress is contained within `buffer`, except for byte compressions
        shape (Tuple[int], Optional): Desired shape of decompressed object. Reshape will attempt to match this shape before returning.
        dtype (str, Optional): Applicable only for byte compressions. Expected dtype of decompressed array.
        compression (str, Optional): Applicable only for byte compressions. Compression used to compression the given buffer.

    Raises:
        SampleDecompressionError: If decompression fails.
        ValueError: If dtype and shape are not specified for byte compression.

    Returns:
        np.ndarray: Array from the decompressed buffer.
    """
    if compression and get_compression_type(compression) == "byte":
        if dtype is None or shape is None:
            raise ValueError("dtype and shape must be specified for byte compressions.")
        try:
            decompressed_bytes = decompress_bytes(buffer, compression)
            return np.frombuffer(decompressed_bytes, dtype=dtype).reshape(shape)
        except Exception:
            raise SampleDecompressionError()
    try:
        img = Image.open(BytesIO(buffer))
        arr = np.array(img)
        if shape is not None:
            arr = arr.reshape(shape)
        return arr
    except Exception:
        raise SampleDecompressionError()


def _get_bounding_shape(shapes: Sequence[Tuple[int]]) -> Tuple[int, int, int]:
    if len(shapes) == 0:
        return (0, 0, 0)
    channels_shape = shapes[0][2:]
    for shape in shapes:
        if shape[2:] != channels_shape:
            raise ValueError()
    return (max(s[0] for s in shapes), sum(s[1] for s in shapes)) + channels_shape  # type: ignore


def compress_multiple(arrays: Sequence[np.ndarray], compression: str) -> bytes:
    """Compress multiple arrays of different shapes into a single buffer. Used for chunk wise compression."""
    dtype = arrays[0].dtype
    for arr in arrays:
        if arr.dtype != dtype:
            raise SampleCompressionError(
                [arr.shape for shape in arr],  # type: ignore
                compression,
                message="All arrays expected to have same dtype.",
            )
    if get_compression_type(compression) == "byte":
        return compress_bytes(
            b"".join(arr.tobytes() for arr in arrays), compression
        )  # Note: shape and dtype info not included
    canvas = np.zeros(_get_bounding_shape([arr.shape for arr in arrays]), dtype=dtype)
    next_x = 0
    for arr in arrays:
        canvas[: arr.shape[0], next_x : next_x + arr.shape[1]] = arr
        next_x += arr.shape[1]
    return compress_array(canvas, compression=compression)


def decompress_multiple(
    buffer: Union[bytes, memoryview],
    shapes: Sequence[Tuple[int, ...]],
    dtype: Optional[str] = None,
    compression: Optional[str] = None,
) -> List[np.ndarray]:
    """Unpack a compressed buffer into multiple arrays."""
    if compression and get_compression_type(compression) == "byte":
        decompressed_buffer = memoryview(decompress_bytes(buffer, compression))
        arrays = []
        itemsize = np.dtype(dtype).itemsize
        for shape in shapes:
            nbytes = int(np.prod(shape) * itemsize)
            arrays.append(
                np.frombuffer(decompressed_buffer[:nbytes], dtype=dtype).reshape(shape)
            )
            decompressed_buffer = decompressed_buffer[nbytes:]
        return arrays
    canvas = decompress_array(buffer)
    arrays = []
    next_x = 0
    for shape in shapes:
        arrays.append(canvas[: shape[0], next_x : next_x + shape[1]])
        next_x += shape[1]
    return arrays


def verify_compressed_file(path: str, compression: str):
    """Verify the contents of an image file
    Args:
        path (str): Path to the image file
        compression (str): Expected compression of the image file
    """
    try:
        if compression == "png":
            _verify_png(path)
        elif compression == "jpeg":
            _verify_jpeg(path)
        else:
            _fast_decompress(path)
    except Exception as e:
        raise CorruptedSampleError(compression)


def _verify_png(path):
    img = Image.open(path)
    img.verify()


def _verify_jpeg(path):
    # See: https://dev.exiv2.org/projects/exiv2/wiki/The_Metadata_in_JPEG_files#2-The-metadata-structure-in-JPEG
    with open(path, "r+b") as f:
        mm = mmap.mmap(f.fileno(), 0)
        soi = f.read(2)
        # Start of Image
        assert soi == b"\xff\xd8"

        # Look for Baseline DCT marker
        sof_idx = mm.find(b"\xff\xc0", 2)
        if sof_idx == -1:
            # Look for Progressive DCT marker
            sof_idx = mm.find(b"\xff\xc2", 2)
            if sof_idx == -1:
                raise Exception()  # Caught by verify_compressed_file()
        f.seek(sof_idx + 2)
        length = int.from_bytes(f.read(2), "big")
        f.seek(length - 2, 1)
        definition_start = f.read(2)
        assert definition_start in [
            b"\xff\xc4",
            b"\xff\xdb",
            b"\xff\xdd",
        ]  # DHT, DQT, DRI

        # TODO this check is too slow
        assert mm.find(b"\xff\xd9") != -1  # End of Image


def _fast_decompress(path):
    img = Image.open(path)
    img.load()
    if img.mode == 1:
        args = ("L",)
    else:
        args = (img.mode,)
    enc = Image._getencoder(img.mode, "raw", args)
    enc.setimage(img.im)
    bufsize = max(65536, img.size[0] * 4)
    while True:
        status, err_code, buf = enc.encode(
            bufsize
        )  # See https://github.com/python-pillow/Pillow/blob/master/src/encode.c#L144
        if err_code:
            break
    if err_code < 0:
        raise Exception()  # caught by verify_compressed_file()
