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
import struct
import sys


if sys.byteorder == "little":
    _NATIVE_INT32 = "<i4"
    _NATIVE_FLOAT32 = "<f4"
else:
    _NATIVE_INT32 = ">i4"
    _NATIVE_FLOAT32 = ">f4"

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
        decompress_array(compressed_bytes, array.shape)
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
    """Gets the shape of a bounding box that can contain the given the shapes tiled horizontally."""
    if len(shapes) == 0:
        return (0, 0, 0)
    channels_shape = shapes[0][2:]
    for shape in shapes:
        if shape[2:] != channels_shape:
            raise ValueError()
    return (max(s[0] for s in shapes), sum(s[1] for s in shapes)) + channels_shape  # type: ignore


def compress_multiple(arrays: Sequence[np.ndarray], compression: str) -> bytes:
    """Compress multiple arrays of different shapes into a single buffer. Used for chunk wise compression.
    The arrays are tiled horizontally and padded with zeros to fit in a bounding box, which is then compressed."""
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


def verify_compressed_file(file, compression: str):
    """Verify the contents of an image file
    Args:
        path (str): Path to the image file
        compression (str): Expected compression of the image file
    """
    if isinstance(file, str):
        f = open(file, "rb")
        close = True
    elif hasattr(file, "read"):
        f = file
        close = False
        f.seek(0)
    else:
        f = file
        close = False
    try:
        if compression == "png":
            return _verify_png(f)
        elif compression == "jpeg":
            return "uint8", _verify_jpeg(f)
        else:
            return _fast_decompress(f)
    except Exception as e:
        raise CorruptedSampleError(compression)
    finally:
        if close:
            f.close()


def get_compression(header):
    if not Image.OPEN:
        Image.init()
    for fmt in Image.OPEN:
        accept = Image.OPEN[fmt][1]
        if accept and accept(header):
            return fmt.lower()
    raise SampleDecompressionError()


def _verify_png(buf):
    if not hasattr(buf, "read"):
        buf = BytesIO(buf)
    img = Image.open(buf)
    img.verify()
    return Image._conv_type_shape(img)


def _verify_jpeg(f):
    if hasattr(f, "read"):
        return _verify_jpeg_file(f)
    return _verify_jpeg_buffer(f)


def _verify_jpeg_buffer(buf: bytes):
    # Start of Image
    mview = memoryview(buf)
    assert buf.startswith(b"\xff\xd8")
    # Look for Baseline DCT marker
    sof_idx = buf.find(b"\xff\xc0", 2)
    if sof_idx == -1:
        # Look for Progressive DCT marker
        sof_idx = buf.find(b"\xff\xc2", 2)
        if sof_idx == -1:
            raise Exception()
    length = int.from_bytes(mview[sof_idx + 2 : sof_idx + 4], "big")
    assert mview[sof_idx + length + 2 : sof_idx + length + 4] in [
        b"\xff\xc4",
        b"\xff\xdb",
        b"\xff\xdd",
    ]  # DHT, DQT, DRI
    shape = struct.unpack(">HHB", mview[sof_idx + 5 : sof_idx + 10])
    assert buf.find(b"\xff\xd9") != -1
    return shape


def _verify_jpeg_file(f):
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
        f.seek(sof_idx + 5)
        shape = struct.unpack(">HHB", f.read(5))
        # TODO this check is too slow
        assert mm.find(b"\xff\xd9") != -1  # End of Image
        return shape
    finally:
        mm.close()


def _fast_decompress(buf):
    if not hasattr(buf, "read"):
        buf = BytesIO(buf)
    img = Image.open(buf)
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
    return Image._conv_type_shape(img)


def read_meta_from_compressed_file(
    file, compression: Optional[str] = None
) -> Tuple[str, Tuple[int], str]:
    """Reads shape, dtype and format without decompressing or verifying the sample."""
    if isinstance(file, str):
        f = open(file, "rb")
        close = True
    elif hasattr(file, "read"):
        f = file
        close = False
        f.seek(0)
    else:
        f = file
        close = False
    try:
        if compression is None:
            if hasattr(f, "read"):
                compression = get_compression(f.read(32))
                f.seek(0)
            else:
                compression = get_compression(f[:32])  # type: ignore
        if compression == "jpeg":
            try:
                shape, typestr = _read_jpeg_shape(f), "|u1"
            except Exception:
                raise CorruptedSampleError("jpeg")
        elif compression == "png":
            try:
                shape, typestr = _read_png_shape_and_dtype(f)
            except Exception:
                raise CorruptedSampleError("png")
        else:
            f.seek(0)
            img = Image.open(f)
            shape, typestr = Image._conv_type_shape(img)
            compression = img.format.lower()
        return compression, shape, typestr   # type: ignore
    finally:
        if close:
            f.close()


def _read_jpeg_shape(f) -> Tuple[int]:
    if hasattr(f, "read"):
        return _read_jpeg_shape_from_file(f)
    return _read_jpeg_shape_from_buffer(f)


def _read_jpeg_shape_from_file(f) -> Tuple[int]:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_COPY)
    try:
        sof_idx = mm.find(b"\xff\xc0", 2)
        if sof_idx == -1:
            sof_idx = mm.find(b"\xff\xc2", 2)
            if sof_idx == -1:
                raise Exception()
        f.seek(sof_idx + 5)
        return struct.unpack(">HHB", f.read(5))  # type: ignore
    finally:
        pass
        mm.close()


def _read_jpeg_shape_from_buffer(buf: bytes) -> Tuple[int]:
    sof_idx = buf.find(b"\xff\xc0", 2)
    if sof_idx == -1:
        sof_idx = buf.find(b"\xff\xc2", 2)
        if sof_idx == -1:
            raise Exception()
    return struct.unpack(">HHB", memoryview(buf)[sof_idx + 5 : sof_idx + 10])  # type: ignore


def _read_png_shape_and_dtype(f) -> Tuple[Tuple[int], str]:
    if not hasattr(f, "read"):
        f = BytesIO(f)
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
    return size + (nlayers,), typstr  # type: ignore
