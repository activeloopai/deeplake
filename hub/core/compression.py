from hub.constants import SUPPORTED_COMPRESSIONS
from hub.util.exceptions import (
    SampleCompressionError,
    SampleDecompressionError,
    UnsupportedCompressionError,
)
from typing import Union, Tuple
import numpy as np

from PIL import Image, UnidentifiedImageError  # type: ignore
from io import BytesIO


def to_image(array: np.ndarray) -> Image:
    shape = array.shape
    if len(shape) == 3 and shape[0] != 1 and shape[2] == 1:
        # convert (X,Y,1) grayscale to (X,Y) for pillow compatibility
        return Image.fromarray(array.squeeze(axis=2))

    return Image.fromarray(array)


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

    if compression not in SUPPORTED_COMPRESSIONS:
        raise UnsupportedCompressionError(compression)

    if compression is None:
        return array.tobytes()

    try:
        img = to_image(array)
        out = BytesIO()
        img.save(out, compression)
        out.seek(0)
        return out.read()
    except (TypeError, OSError) as e:
        raise SampleCompressionError(array.shape, compression, str(e))


def decompress_array(buffer: Union[bytes, memoryview], shape: Tuple[int]) -> np.ndarray:
    """Decompress some buffer into a numpy array. It is expected that all meta information is
    stored inside `buffer`.

    Note:
        `compress_array` may be used to get the `buffer` input.

    Args:
        buffer (bytes, memoryview): Buffer to be decompressed. It is assumed all meta information required to
            decompress is contained within `buffer`.
        shape (Tuple[int]): Desired shape of decompressed object. Reshape will attempt to match this shape before returning.

    Raises:
        SampleDecompressionError: Right now only buffers compatible with `PIL` will be decompressed.

    Returns:
        np.ndarray: Array from the decompressed buffer.
    """

    try:
        img = Image.open(BytesIO(buffer))
        return np.array(img).reshape(shape)
    except UnidentifiedImageError:
        raise SampleDecompressionError()
