from hub.constants import SUPPORTED_COMPRESSIONS
from hub.util.exceptions import (
    SampleCompressionError,
    SampleDecompressionError,
    UnsupportedCompressionError,
)
from typing import Union, Tuple, Sequence, List, Optional
import numpy as np

from PIL import Image, UnidentifiedImageError  # type: ignore
from io import BytesIO

import lz4.frame


np_id2dtype = {}

for dtype in np.cast:
    dtype = np.dtype(dtype)
    np_id2dtype[dtype.num] = dtype


def to_image(array: np.ndarray) -> Image:
    shape = array.shape
    if len(shape) == 3 and shape[0] != 1 and shape[2] == 1:
        # convert (X,Y,1) grayscale to (X,Y) for pillow compatibility
        return Image.fromarray(array.squeeze(axis=2))

    return Image.fromarray(array)


_LZ4_HEADER = b"lz4!"


def _lz4_compress_array(array: np.ndarray) -> bytes:
    shape_bytes = b"".join(d.to_bytes(4, "little") for d in array.shape)
    ndim_byte = array.ndim.to_bytes(1, "little")
    dtype_byte = array.dtype.num.to_bytes(1, "little")
    return b"".join([_LZ4_HEADER, dtype_byte, ndim_byte, shape_bytes, array.tobytes()])


def _lz4_decompress_array(buffer: Union[bytes, memoryview]) -> np.ndarray:
    if isinstance(buffer, bytes):
        buffer = memoryview(buffer)
    buffer = buffer[len(_LZ4_HEADER) :]
    dtype = np_id2dtype[int.from_bytes(buffer[:1], "little")]
    ndim = int.from_bytes(buffer[1:2], "little")
    shape_bytes = buffer[2 : 2 + ndim * 4]
    shape = tuple(
        int.from_bytes(shape_bytes[i * 4 : i * 4 + 4], "little") for i in range(ndim)
    )
    return (
        np.frombuffer(buffer[ndim * 4 + 2 :], dtype=np.byte).view(dtype).reshape(shape)
    )


def compress_array(array: np.ndarray, compression: str) -> bytes:
    """Compress some numpy array using `compression`. All meta information will be contained in the returned buffer.

    Note:
        `decompress_array` may be used to decompress from the returned bytes back into the `array`.

    Args:
        array (np.ndarray): Array to be compressed.
        compression (str): `array` will be compressed with this compression into bytes. Right now only arrays compatible with `PIL` will be compressed.

    Raises:
        UnsupportedCompressionError: If `compression` is unsupported. See `SUPPORTED_COMPRESSIONS`.
        SampleCompressionError: If there was a problem compressing `array`.

    Returns:
        bytes: Compressed `array` represented as bytes.
    """

    # empty sample shouldn't be compressed

    if 0 in array.shape:
        return bytes()

    if compression not in SUPPORTED_COMPRESSIONS:
        raise UnsupportedCompressionError(compression)

    if compression is None:
        return array.tobytes()

    if compression == "lz4":
        return _lz4_compress_array(array)

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
    buffer: Union[bytes, memoryview], shape: Optional[Tuple[int]] = None
) -> np.ndarray:
    """Decompress some buffer into a numpy array. It is expected that all meta information is
    stored inside `buffer`.

    Note:
        `compress_array` may be used to get the `buffer` input.

    Args:
        buffer (bytes, memoryview): Buffer to be decompressed. It is assumed all meta information required to
            decompress is contained within `buffer`, except for lz4
        shape (Tuple[int]): Desired shape of decompressed object. Reshape will attempt to match this shape before returning.
    Raises:
        SampleDecompressionError: Right now only buffers compatible with `PIL` will be decompressed.

    Returns:
        np.ndarray: Array from the decompressed buffer.
    """

    if memoryview(buffer)[: len(_LZ4_HEADER)] == _LZ4_HEADER:
        return _lz4_decompress_array(buffer)
    try:
        img = Image.open(BytesIO(buffer))
        arr = np.array(img)
        if shape is not None:
            arr = arr.reshape(shape)
        return arr
    except UnidentifiedImageError:
        raise SampleDecompressionError()


def _get_bounding_shape(shapes: Sequence[Tuple[int]]) -> Tuple[int]:
    if len(shapes) == 0:
        return (0, 0)
    channels_shape = shapes[0][2:]
    for shape in shapes:
        if shape[2:] != channels_shape:
            raise ValueError()
    return (max(s[0] for s in shapes), sum(s[1] for s in shapes)) + channels_shape


def compress_multiple(arrays: Sequence[np.ndarray], compression: str) -> bytes:
    dtype = arrays[0].dtype
    for arr in arrays:
        if arr.dtype != dtype:
            raise TypeError()  # TODO
    if compression == "lz4":
        return lz4.frame.compress(
            b"".join(arr.tobytes() for arr in arrays)
        )  # Note: shape and dtype info not included
    canvas = np.zeros(_get_bounding_shape([arr.shape for arr in arrays]), dtype=dtype)
    next_x = 0
    for arr in arrays:
        canvas[: arr.shape[0], next_x : next_x + arr.shape[1]] = arr
        next_x += arr.shape[1]
    return compress_array(canvas, compression=compression)


def decompress_multiple(
    buffer: Union[bytes, memoryview], shapes: Sequence[Tuple[int]]
) -> List[np.ndarray]:
    canvas = decompress_array(buffer)
    arrays = []
    next_x = 0
    for shape in shapes:
        arrays.append(canvas[: shape[0], next_x : next_x + shape[1]])
        next_x += shape[1]
    return arrays
