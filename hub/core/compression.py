import hub
from hub.core._compression import NATIVE_INT32, STRUCT_II
from hub.core._compression import JPEG, PNG
from hub.util.compression import re_find_first
from hub.util.exceptions import (
    SampleCompressionError,
    SampleDecompressionError,
    UnsupportedCompressionError,
    CorruptedSampleError,
)
from hub.compression import (
    get_compression_type,
    BYTE_COMPRESSION,
    IMAGE_COMPRESSION,
    AUDIO_COMPRESSION,
)
from typing import Union, Tuple, Sequence, List, Optional, BinaryIO
import numpy as np

from PIL import Image, UnidentifiedImageError  # type: ignore
from io import BytesIO
import mmap
import struct
import sys
import re
import numcodecs.lz4  # type: ignore
import lz4.frame  # type: ignore
import os
import tempfile
from miniaudio import mp3_read_file_f32, mp3_read_f32, mp3_get_file_info, mp3_get_info  # type: ignore


def to_image(array: np.ndarray) -> Image:
    shape = array.shape
    if len(shape) == 3 and shape[0] != 1 and shape[2] == 1:
        # convert (X,Y,1) grayscale to (X,Y) for pillow compatibility
        return Image.fromarray(array.squeeze(axis=2))

    return Image.fromarray(array)


def compress_bytes(buffer: Union[bytes, memoryview], compression: str) -> bytes:
    if compression == "lz4":
        return numcodecs.lz4.compress(buffer)
    else:
        raise SampleCompressionError(
            (len(buffer),), compression, f"Not a byte compression: {compression}"
        )


def decompress_bytes(buffer: Union[bytes, memoryview], compression: str) -> bytes:
    if compression == "lz4":
        if (
            buffer[:4] == b'\x04"M\x18'
        ):  # python-lz4 magic number (backward compatiblity)
            return lz4.frame.decompress(buffer)
        return numcodecs.lz4.decompress(buffer)
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
        NotImplementedError: If compression is not supported.

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

    compr_type = get_compression_type(compression)

    if compr_type == BYTE_COMPRESSION:
        return compress_bytes(array.tobytes(), compression)
    elif compr_type == AUDIO_COMPRESSION:
        raise NotImplementedError(
            "In order to store audio data, you should use `hub.read(path_to_file)`. Compressing raw data is not yet supported."
        )
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
    buffer: Union[bytes, memoryview, str],
    shape: Optional[Tuple[int]] = None,
    dtype: Optional[str] = None,
    compression: Optional[str] = None,
) -> np.ndarray:
    """Decompress some buffer into a numpy array. It is expected that all meta information is
    stored inside `buffer`.

    Note:
        `compress_array` may be used to get the `buffer` input.

    Args:
        buffer (bytes, memoryview, str): Buffer or file to be decompressed. It is assumed all meta information required to
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
    compr_type = get_compression_type(compression)
    if compr_type == BYTE_COMPRESSION:
        if dtype is None or shape is None:
            raise ValueError("dtype and shape must be specified for byte compressions.")
        try:
            decompressed_bytes = decompress_bytes(buffer, compression)  # type: ignore
            return np.frombuffer(decompressed_bytes, dtype=dtype).reshape(shape)
        except Exception:
            raise SampleDecompressionError()
    elif compr_type == AUDIO_COMPRESSION:
        return _decompress_mp3(buffer)
    try:
        if not isinstance(buffer, str):
            buffer = BytesIO(buffer)  # type: ignore
        img = Image.open(buffer)  # type: ignore
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
    compr_type = get_compression_type(compression)
    if compr_type == BYTE_COMPRESSION:
        return compress_bytes(
            b"".join(arr.tobytes() for arr in arrays), compression
        )  # Note: shape and dtype info not included
    elif compr_type == AUDIO_COMPRESSION:
        raise NotImplementedError("compress_multiple does not support audio samples.")
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


def verify_compressed_file(
    file: Union[str, BinaryIO, bytes], compression: str
) -> Tuple[Tuple[int, ...], str]:
    """Verify the contents of an image file
    Args:
        file (Union[str, BinaryIO]): Path to the file or file like object or contents of the file
        compression (str): Expected compression of the image file
    """
    if isinstance(file, str):
        file = open(file, "rb")
        close = True
    elif hasattr(file, "read"):
        close = False
        file.seek(0)  # type: ignore
    else:
        close = False
    try:
        if compression == "png":
            return PNG(file).verify()
        elif compression == "jpeg":
            return JPEG(file).verify()
        elif compression == "mp3":
            return _read_mp3_shape(file), "<f4"  # type: ignore
        else:
            return _fast_decompress(file)
    except Exception as e:
        raise CorruptedSampleError(compression)
    finally:
        if close:
            file.close()  # type: ignore


def get_compression(header=None, path=None):
    if path:
        if path.lower().endswith(".mp3"):
            return "mp3"
    if header:
        if not Image.OPEN:
            Image.init()
        for fmt in Image.OPEN:
            accept = Image.OPEN[fmt][1]
            if accept and accept(header):
                return fmt.lower()
        raise SampleDecompressionError()


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
        isfile = True
        close = True
    elif hasattr(file, "read"):
        f = file
        close = False
        isfile = True
        f.seek(0)
    else:
        isfile = False
        f = file
        close = False
    try:
        if compression is None:
            path = file if isinstance(file, str) else None
            if hasattr(f, "read"):
                compression = get_compression(f.read(32), path)
                f.seek(0)
            else:
                compression = get_compression(f[:32], path)  # type: ignore
        if compression == "jpeg":
            try:
                shape, typestr = JPEG(f).read_shape_and_dtype()
            except Exception:
                raise CorruptedSampleError("jpeg")
        elif compression == "png":
            try:
                shape, typestr = _read_png_shape_and_dtype(f)
            except Exception:
                raise CorruptedSampleError("png")
        elif compression == "mp3":
            try:
                shape, typestr = _read_mp3_shape(file), "<f4"
            except Exception as e:
                raise CorruptedSampleError("mp3")
        else:
            img = Image.open(f) if isfile else Image.open(BytesIO(f))  # type: ignore
            shape, typestr = Image._conv_type_shape(img)
            compression = img.format.lower()
        return compression, shape, typestr  # type: ignore
    finally:
        if close:
            f.close()


def _read_png_shape_and_dtype(f: Union[bytes, BinaryIO]) -> Tuple[Tuple[int, ...], str]:
    """Reads shape and dtype of a png file from a file like object or file contents.
    If a file like object is provided, all of its contents are NOT loaded into memory."""
    if not hasattr(f, "read"):
        f = BytesIO(f)  # type: ignore
    f.seek(16)  # type: ignore
    size = STRUCT_II.unpack(f.read(8))[::-1]  # type: ignore
    bits, colors = f.read(2)  # type: ignore

    # Get the number of channels and dtype based on bits and colors:
    if colors == 0:
        if bits == 1:
            typstr = "|b1"
        elif bits == 16:
            typstr = NATIVE_INT32
        else:
            typstr = "|u1"
        nlayers = None
    else:
        typstr = "|u1"
        if colors == 2:
            nlayers = 3
        elif colors == 3:
            nlayers = None
        elif colors == 4:
            if bits == 8:
                nlayers = 2
            else:
                nlayers = 4
        else:
            nlayers = 4
    shape = size if nlayers is None else size + (nlayers,)
    return shape, typstr  # type: ignore


def _decompress_mp3(file: Union[bytes, memoryview, str]) -> np.ndarray:
    decompressor = mp3_read_file_f32 if isinstance(file, str) else mp3_read_f32
    if isinstance(file, memoryview):
        if (
            isinstance(file.obj, bytes)
            and file.strides == (1,)
            and file.shape == (len(file.obj),)
        ):
            file = file.obj
        else:
            file = bytes(file)
    raw_audio = decompressor(file)
    return np.frombuffer(raw_audio.samples, dtype="<f4").reshape(
        raw_audio.num_frames, raw_audio.nchannels
    )


def _read_mp3_shape(file: Union[bytes, memoryview, str]) -> Tuple[int, ...]:
    f_info = mp3_get_file_info if isinstance(file, str) else mp3_get_info
    info = f_info(file)
    return (info.num_frames, info.nchannels)
