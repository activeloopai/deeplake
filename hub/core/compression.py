from hub.constants import SUPPORTED_COMPRESSIONS
from hub.util.exceptions import (
    SampleCompressionError,
    SampleDecompressionError,
    UnsupportedCompressionError,
    CorruptedSampleError,
)
from typing import Union, Tuple
import numpy as np

from PIL import Image, UnidentifiedImageError  # type: ignore
from io import BytesIO
import mmap
import struct
import sys


if sys.byteorder == "little":
    _NATIVE_INT32 = "<i4"
    _NATIVE_FLOAT32 = "<f4"
else:
    _NATIVE_INT32 = ">i4"
    _NATIVE_FLOAT32 = ">f4"


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

    # empty sample shouldn't be compressed
    if 0 in array.shape:
        return bytes()

    if compression not in SUPPORTED_COMPRESSIONS:
        raise UnsupportedCompressionError(compression)

    if compression is None:
        return array.tobytes()

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

    img = Image.open(BytesIO(buffer))
    print(img)
    return np.array(img).reshape(shape)


def verify_compressed_file(file, compression: str):
    """Verify the contents of an image file
    Args:
        path (str): Path to the image file
        compression (str): Expected compression of the image file
    """
    if isinstance(file, str):
        f = open(file, "rb")
        close = True
    else:
        f = file
        close = False
        f.seek(0)
    try:
        if compression == "png":
            return _verify_png(f)
        elif compression == "jpeg":
            return _verify_jpeg(f)
        else:
            return _fast_decompress(f)
    except Exception as e:
        raise CorruptedSampleError(compression)
    finally:
        if close:
            f.close()


def _verify_png(f):
    img = Image.open(f)
    img.verify()


def _verify_jpeg(f):
    # See: https://dev.exiv2.org/projects/exiv2/wiki/The_Metadata_in_JPEG_files#2-The-metadata-structure-in-JPEG
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    try:
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
    finally:
        mm.close()


def _fast_decompress(f):
    img = Image.open(f)
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


def read_meta_from_compressed_file(file) -> Tuple[str, Tuple[int], str]:
    """Reads shape, dtype and format without decompressing or verifying the sample."""
    if isinstance(file, str):
        f = open(file, "rb")
        close = True
    else:
        f = file
        close = False
        f.seek(0)
    try:
        header = f.read(8)
        if header.startswith(b"\xFF\xD8\xFF"):
            try:
                compression, shape, typestr = "jpeg", _read_jpeg_shape(f), "|u1"
            except Exception:
                raise CorruptedSampleError("jpeg")
        elif header.startswith(b"\211PNG\r\n\032\n"):
            try:
                compression, (shape, typestr) = "png", _read_png_shape_and_dtype(f)
            except Exception:
                raise CorruptedSampleError("png")
        else:
            f.seek(0)
            img = Image.open(f)
            shape, typestr = Image._conv_type_shape(img)
            compression = img.format.lower()
        return compression, shape, typestr
    finally:
        if close:
            f.close()


def _read_jpeg_shape(f) -> Tuple[int]:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    try:
        sof_idx = mm.find(b"\xff\xc0", 2)
        if sof_idx == -1:
            sof_idx = mm.find(b"\xff\xc2", 2)
            if sof_idx == -1:
                raise Exception()
        f.seek(sof_idx + 5)
        return struct.unpack(">HHB", f.read(5))
    finally:
        mm.close()


def _read_png_shape_and_dtype(f) -> Tuple[Tuple[int], str]:
    f.seek(16)
    size = struct.unpack(">ii", f.read(8))[::-1]
    im_mode, im_rawmode = f.read(2)
    if im_rawmode == 0:
        if im_mode == 1:
            typstr = "|b1"
        elif im_mode == 16:
            typstr = _NATIVE_INT32
        else:
            typstr = "|u1"
        nlayers = None
    else:
        typstr = "|u1"
        if im_rawmode == 2:
            nlayers = 3
        elif im_rawmode == 3:
            nlayers = None
        elif im_rawmode == 4:
            if im_mode == 8:
                nlayers = 2
            else:
                nlayers = 4
        else:
            nlayers = 4
    return size + (nlayers,), typstr
