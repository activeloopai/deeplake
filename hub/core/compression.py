import warnings
from hub.core.meta.tensor_meta import TensorMeta
import hub
from hub.util.exceptions import (
    SampleCompressionError,
    SampleDecompressionError,
    UnsupportedCompressionError,
    CorruptedSampleError,
)
from hub.compression import (
    get_compression_type,
    IMAGE_COMPRESSION,
    VIDEO_COMPRESSION,
    BYTE_COMPRESSION,
)
from typing import Union, Tuple, Sequence, List, Optional, BinaryIO, Iterator
import numpy as np

from PIL import Image  # type: ignore
from io import BytesIO
import mmap
import math
import struct
import sys
import re
import lz4.frame  # type: ignore
import tempfile
import os


def _import_moviepy():
    global moviepy
    global VideoFileClip
    global concatenate_videoclips
    global crop_video
    import moviepy
    from moviepy.editor import VideoFileClip, concatenate_videoclips
    from moviepy.video.fx.all import crop as crop_video

# this maps a compressor to the average compression ratio it achieves assuming the data is natural
# for example, if compressor X on average removes 50% of the data, the compression factor would be 2.0.
# TODO: for every compressor we have, we should have an accurate number here!
# NOTE: these may need to be tuned, possibly even determined based on htype or user dataset metrics
COMPRESSION_FACTORS = {
    "png": 2.6,
    "jpeg": 18,
    "jpeg2000": 2.3,
    "lz4": 1.3,
    "webp": 26,
}


if sys.byteorder == "little":
    _NATIVE_INT32 = "<i4"
    _NATIVE_FLOAT32 = "<f4"
else:
    _NATIVE_INT32 = ">i4"
    _NATIVE_FLOAT32 = ">f4"


_JPEG_SOFS = [
    b"\xff\xc0",
    b"\xff\xc2",
    b"\xff\xc1",
    b"\xff\xc3",
    b"\xff\xc5",
    b"\xff\xc6",
    b"\xff\xc7",
    b"\xff\xc9",
    b"\xff\xca",
    b"\xff\xcb",
    b"\xff\xcd",
    b"\xff\xce",
    b"\xff\xcf",
    b"\xff\xde",
]

_JPEG_SOFS_RE = re.compile(b"|".join(_JPEG_SOFS))
_STRUCT_HHB = struct.Struct(">HHB")
_STRUCT_II = struct.Struct(">ii")
_STRUCT_F = struct.Struct("f")


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
    if len(buffer) <= 0:
        return bytes()

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

    compr_typ = get_compression_type(compression)

    if compr_typ == BYTE_COMPRESSION:
        return compress_bytes(array.tobytes(), compression)
    elif compr_typ == VIDEO_COMPRESSION:
        raise NotImplementedError(
            "Compressing numpy arrays to videos is not yet supported"
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


def _decompress_mp4(file: Union[bytes, memoryview, str]) -> np.ndarray:
    if isinstance(file, str):
        clip = VideoFileClip(file)
        arr = np.concatenate([frame for frame in clip.iter_frames()])
        clip.close()
        return arr
    else:
        buffer = file
        with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            pass
        with open(f.name, "wb") as fw:
            fw.write(buffer)
            clip = VideoFileClip(f.name)
            arr = np.stack(
                [frame for frame in clip.iter_frames()]
            )
            clip.close()
            return arr


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
        buffer (bytes, memoryview, str): Buffer (or file) to be decompressed. It is assumed all meta information required to
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

    if len(buffer) <= 0:
        return np.zeros(shape, dtype=dtype)

    if compression and get_compression_type(compression) == BYTE_COMPRESSION:
        if dtype is None or shape is None:
            raise ValueError("dtype and shape must be specified for byte compressions.")
        if isinstance(buffer, str):
            with open(buffer, "rb") as f:
                buffer = f.read()
        try:
            decompressed_bytes = decompress_bytes(buffer, compression)
            return np.frombuffer(decompressed_bytes, dtype=dtype).reshape(shape)
        except Exception:
            raise SampleDecompressionError()
    if not compression and _is_mp4(buffer):
        compression = "mp4"
    if compression == "mp4":
        _import_moviepy()
        try:
            return _decompress_mp4(buffer)
        except Exception:
            raise SampleDecompressionError()
    try:
        if not isinstance(buffer, str):
            buffer = BytesIO(buffer)
        img = Image.open(buffer)
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


def compress_multiple(
    samples: Sequence[Union[np.ndarray, str]], compression: str
) -> bytes:
    """Compress multiple arrays of different shapes into a single buffer. Used for chunk wise compression.
    The arrays are tiled horizontally and padded with zeros to fit in a bounding box, which is then compressed."""
    if get_compression_type(compression) == VIDEO_COMPRESSION:
        return _pack_videos(samples)
    arrays = samples
    dtype = arrays[0].dtype
    for arr in arrays:
        if arr.dtype != dtype:
            raise SampleCompressionError(
                [arr.shape for shape in arr],  # type: ignore
                compression,
                message="All arrays expected to have same dtype.",
            )
    if get_compression_type(compression) == BYTE_COMPRESSION:
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
    shapes: Optional[Sequence[Tuple[int, ...]]] = None,
    dtype: Optional[str] = None,
    compression: Optional[str] = None,
) -> List[np.ndarray]:
    """Unpack a compressed buffer into multiple arrays."""
    
    compr_typ = get_compression_type(compression)
    if compr_typ == VIDEO_COMPRESSION:
        return _unpack_videos(buffer)
    elif compr_typ == BYTE_COMPRESSION:
        decompressed_buffer = memoryview(decompress_bytes(buffer, compression))
        arrays = []
        itemsize = np.dtype(dtype).itemsize
        for shape in shapes:
            nbytes = int(np.prod(shape) * itemsize)

            if len(buffer) <= 0:
                # empty array (maybe a tiled sample)
                array = np.zeros(shape, dtype=dtype)
            else:
                array = np.frombuffer(decompressed_buffer[:nbytes], dtype=dtype).reshape(shape)

            arrays.append(array)
            decompressed_buffer = decompressed_buffer[nbytes:]

        return arrays

    if len(buffer) <= 0:
        arrays = []
        for shape in shapes:
            arrays.append(np.zeros(shape, dtype=dtype))
        return arrays

    canvas = decompress_array(buffer)
    arrays = []
    next_x = 0
    for shape in shapes:
        arrays.append(canvas[: shape[0], next_x : next_x + shape[1]])
        next_x += shape[1]
    return arrays


def _pack_videos(paths: Sequence[str]) -> memoryview:
    _import_moviepy()
    with tempfile.TemporaryFile(suffix=".mp4") as f:
        fname = f.name
    clips = [VideoFileClip(path) for path in paths]
    sizes = [clip.size for clip in clips]
    lengths = [clip.duration for clip in clips]
    try:
        concatenate_videoclips(clips).write_videofile(fname)
        with open(fname, "rb") as f:
            byts = f.read()
    finally:
        if os.path.isfile(fname):
            os.remove(fname)
    [clip.close() for clip in clips]
    header_size = 2 + 4 * 2 * len(clips) + 4 * len(clips)
    ba = bytearray(header_size + len(byts))
    ba[:2] = len(clips).to_bytes(2, "big")
    offset = 2
    for size, length in zip(sizes, lengths):
        ba[offset : offset + 8] = _STRUCT_II.pack(*size)
        offset += 8
        ba[offset : offset + 4] = _STRUCT_F.pack(length)
        offset += 4
    ba[offset:] = byts
    return memoryview(ba)


def _unpack_videos(buffer) -> List[np.ndarray]:
    _import_moviepy()
    buffer = memoryview(buffer)
    nclips = int.from_bytes(buffer[:2], "big")
    offset = 2
    sizes = []
    lengths = []
    for i in range(nclips):
        size = _STRUCT_II.unpack(buffer[offset : offset + 8])
        offset += 8
        length = _STRUCT_F.unpack(buffer[offset : offset + 4])[0]
        offset += 4
        sizes.append(size)
        lengths.append(length)
    subclips = []
    with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
        fname = f.name
    try:
        with open(fname, "wb") as f:
            f.write(buffer[offset:])
        clip = VideoFileClip(f.name)
        x_c = clip.w // 2
        y_c = clip.h // 2
        offset = 0
        for size, length in zip(sizes, lengths):
            subclip = clip.subclip(offset, offset + length)
            offset += length
            subclip = crop_video(
                subclip, x_center=x_c, y_center=y_c, width=size[0], height=size[1]
            )
            subclips.append(subclip)

        return [
            np.stack([frame for frame in subclip.iter_frames()])
            for subclip in subclips
        ]
    finally:
        clip.close()
        if os.path.isfile(fname):
            os.remove(fname)

def verify_compressed_file(
    file: Union[str, BinaryIO, bytes], compression: str
) -> Tuple[Tuple[int, ...], str]:
    """Verify the contents of an image file
    Args:
        file (Union[str, BinaryIO]): Path to the file or file like object or contents of the file
        compression (str): Expected compression of the image file
    """
    if isinstance(file, str):
        f = open(file, "rb")
        close = True
    elif hasattr(file, "read"):
        f = file
        close = False
        file.seek(0)  # type: ignore
    else:
        f = file
        close = False
    try:
        if compression == "png":
            return _verify_png(f)
        elif compression == "jpeg":
            return _verify_jpeg(f), "|u1"
        elif compression == "mp4":
            return _read_mp4_shape(file), "|u1"
        else:
            return _fast_decompress(f)
    except Exception as e:
        raise CorruptedSampleError(compression)
    finally:
        if close:
            f.close()  # type: ignore


_MP4_CHUNK_SUBTYPES = [
    "avc1",
    "iso2",
    "isom",
    "mmp4",
    "mp41",
    "mp42",
    "mp71",
    "msnv",
    "ndas",
    "ndsc",
    "ndsh",
    "ndsm",
    "ndsp",
    "ndss",
    "ndxc",
    "ndxh",
    "ndxm",
    "ndxp",
    "ndxs",
]

_MP4_CHUNK_SUBTYPES = [typ.encode("ascii") for typ in _MP4_CHUNK_SUBTYPES]


def _is_mp4(header: bytes) -> bool:
    header = memoryview(header)
    return header[4:8] == b"ftyp" and header[8:12] in _MP4_CHUNK_SUBTYPES


def split_video(path: str, chunk_size: int, return_shapes: bool = False) -> Iterator[bytes]:
    _import_moviepy()
    clip = VideoFileClip(path)
    clip_nbytes = os.path.getsize(path)
    if clip_nbytes <= chunk_size:
        with open(path, "rb") as f:
            if return_shapes:
                yield (f.read(), (math.ceil(clip.duration * clip.fps), ) + tuple(clip.size)[::-1] + (3,))
            else:
                yield f.read()
            clip.close()
            return
    subclip_length = clip.duration * chunk_size / clip_nbytes
    nsubclips = math.ceil(clip_nbytes / chunk_size)
    subclips = [
        clip.subclip(i * subclip_length, min(clip.end, (i + 1) * subclip_length))
        for i in range(nsubclips)
    ]
    ext = os.path.splitext(path)[-1]
    with tempfile.NamedTemporaryFile(suffix=ext) as f:
        fname = f.name
    try:
        for subclip in subclips:
            subclip.write_videofile(fname)
            with open(fname, "rb") as f:
                if return_shapes:
                    yield (f.read(), (math.ceil(subclip.duration * subclip.fps), ) + tuple(subclip.size)[::-1] + (3,))
                else:
                    yield f.read()
    finally:
        clip.close()
        if os.path.isfile(fname):
            os.remove(fname)


def get_compression(header: bytes) -> str:
    if _is_mp4(header):
        return "mp4"
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
    # Look for Start of Frame
    sof_idx = -1
    for sof_match in re.finditer(_JPEG_SOFS_RE, buf):
        sof_idx = sof_match.start(0)
    if sof_idx == -1:
        raise Exception()

    length = int.from_bytes(mview[sof_idx + 2 : sof_idx + 4], "big")
    assert mview[sof_idx + length + 2 : sof_idx + length + 4] in [
        b"\xff\xc4",
        b"\xff\xdb",
        b"\xff\xdd",
    ]  # DHT, DQT, DRI
    shape = _STRUCT_HHB.unpack(mview[sof_idx + 5 : sof_idx + 10])
    assert buf.find(b"\xff\xd9") != -1
    if shape[-1] in (1, None):
        shape = shape[:-1]
    return shape


def _verify_jpeg_file(f):
    # See: https://dev.exiv2.org/projects/exiv2/wiki/The_Metadata_in_JPEG_files#2-The-metadata-structure-in-JPEG
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    try:
        soi = f.read(2)
        # Start of Image
        assert soi == b"\xff\xd8"

        # Look for Start of Frame
        sof_idx = -1
        for sof_match in re.finditer(_JPEG_SOFS_RE, mm):
            sof_idx = sof_match.start(0)
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
        shape = _STRUCT_HHB.unpack(f.read(5))
        # TODO this check is too slow
        assert mm.find(b"\xff\xd9") != -1  # End of Image
        if shape[-1] in (1, None):
            shape = shape[:-1]
        return shape
    finally:
        mm.close()


def _fast_decompress(buf):
    """Slightly faster than `np.array(Image.open(...))`."""

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


def get_compression_factor(tensor_meta: TensorMeta) -> float:
    factor = 1.0

    # check sample compression first. we don't support compressing both sample + chunk-wise at the same time, but in case we
    # do support this in the future, try both.
    sc = tensor_meta.sample_compression

    if sc is not None:
        if sc not in COMPRESSION_FACTORS:
            # TODO: this should ideally never happen
            warnings.warn(f"Warning: the provided compression \"{sc}\" has no approximate factor yet, so you should expect tiles to be inefficient!")
            return factor

        factor *= COMPRESSION_FACTORS[sc]

    # TODO: UNCOMMENT AFTER CHUNK-WISE COMPRESSION IS MERGED!
    # cc = tensor_meta.chunk_compression
    # if cc is not None:
    #     factor *= COMPRESSION_FACTORS[cc]

    return factor


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
        elif compression == "mp4":
            _import_moviepy()
            try:
                shape, typestr = _read_mp4_shape(file), "|u1"
            except Exception:
                raise CorruptedSampleError("mp4")
        else:
            img = Image.open(f) if isfile else Image.open(BytesIO(f))  # type: ignore
            shape, typestr = Image._conv_type_shape(img)
            compression = img.format.lower()
        return compression, shape, typestr  # type: ignore
    finally:
        if close:
            f.close()


def _read_jpeg_shape(f: Union[bytes, BinaryIO]) -> Tuple[int, ...]:
    if hasattr(f, "read"):
        return _read_jpeg_shape_from_file(f)
    return _read_jpeg_shape_from_buffer(f)  # type: ignore


def _read_jpeg_shape_from_file(f) -> Tuple[int, ...]:
    """Reads shape of a jpeg image from file without loading the whole image in memory"""
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_COPY)
    try:
        # Look for Start of Frame
        sof_idx = -1
        for sof_match in re.finditer(_JPEG_SOFS_RE, mm):  # type: ignore
            sof_idx = sof_match.start(0)
        if sof_idx == -1:
            raise Exception()
        f.seek(sof_idx + 5)
        shape = _STRUCT_HHB.unpack(f.read(5))  # type: ignore
        if shape[-1] in (1, None):
            shape = shape[:-1]
        return shape
    finally:
        pass
        mm.close()


def _read_jpeg_shape_from_buffer(buf: bytes) -> Tuple[int, ...]:
    """Gets shape of a jpeg file from its contents"""
    # Look for Start of Frame
    sof_idx = -1
    for sof_match in re.finditer(_JPEG_SOFS_RE, buf):
        sof_idx = sof_match.start(0)
    if sof_idx == -1:
        raise Exception()
    shape = _STRUCT_HHB.unpack(memoryview(buf)[sof_idx + 5 : sof_idx + 10])  # type: ignore
    if shape[-1] in (1, None):
        shape = shape[:-1]
    return shape


def _read_png_shape_and_dtype(f: Union[bytes, BinaryIO]) -> Tuple[Tuple[int, ...], str]:
    """Reads shape and dtype of a png file from a file like object or file contents.
    If a file like object is provided, all of its contents are NOT loaded into memory."""
    if not hasattr(f, "read"):
        f = BytesIO(f)  # type: ignore
    f.seek(16)  # type: ignore
    size = _STRUCT_II.unpack(f.read(8))[::-1]  # type: ignore
    bits, colors = f.read(2)  # type: ignore

    # Get the number of channels and dtype based on bits and colors:
    if colors == 0:
        if bits == 1:
            typstr = "|b1"
        elif bits == 16:
            typstr = _NATIVE_INT32
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


def _read_mp4_shape(f) -> Tuple[int, int, int, int]:
    assert isinstance(f, str), f"Required mp4 file path as str, received {type(f)}"
    clip = VideoFileClip(f)
    nframes = clip.reader.nframes
    return (nframes,) + next(clip.iter_frames()).shape  # TODO: avoid decompression here
