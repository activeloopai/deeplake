from hub.util.exceptions import SampleDecompressionError
from typing import Union
import numpy as np

from PIL import Image, UnidentifiedImageError  # type: ignore
from io import BytesIO


def compress_array(array: np.ndarray, compression: str) -> bytes:
    """Compress some numpy array using `compression`. All meta information will be contained in the returned buffer.

    Args:
        compression (str): Output bytes will be compressed with this compression. Right now only arrays compatible with `PIL` will be compressed.

    Returns:
        bytes: Compressed buffer.
    """

    img = Image.fromarray(array.astype("uint8"))
    out = BytesIO()
    img.save(out, compression)
    out.seek(0)
    return out.read()


def decompress_array(buffer: Union[bytes, memoryview]) -> np.ndarray:
    """Decompress some buffer into a numpy array. It is expected that all meta information is
    stored inside `buffer`.

    Raises:
        BufferDecompressionError: Right now only buffers compatible with `PIL` will be decompressed.

    Returns:
        np.ndarray: Array from the decompressed buffer.
    """

    try:
        img = Image.open(BytesIO(buffer))
        return np.array(img)
    except UnidentifiedImageError:
        raise SampleDecompressionError()
