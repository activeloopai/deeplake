from hub.compression import (
    BYTE_COMPRESSION,
    VIDEO_COMPRESSION,
    AUDIO_COMPRESSION,
    get_compression_type,
)
from hub.core.fast_forwarding import version_compare
from hub.core.tiling.sample_tiles import SampleTiles
from hub.core.partial_sample import PartialSample
from hub.util.compression import get_compression_ratio  # type: ignore
from hub.util.exceptions import TensorInvalidSampleShapeError
from hub.util.casting import intelligent_cast
from hub.util.json import HubJsonDecoder, HubJsonEncoder, validate_json_object
from hub.core.sample import Sample, SampleValue  # type: ignore
from hub.core.compression import compress_array, compress_bytes
from typing import Optional, Sequence, Union, Tuple
import hub
import numpy as np
import struct
import json
from urllib.request import Request, urlopen

BaseTypes = Union[np.ndarray, list, int, float, bool, np.integer, np.floating, np.bool_]
HEADER_SIZE_BYTES = 13


def infer_header_num_bytes(
    version: str, shape_info: np.ndarray, byte_positions: np.ndarray
):
    """Calculates the number of header bytes in a chunk without serializing it.

    Args:
        version: (str) Version of hub library
        shape_info: (numpy.ndarray) Encoded shapes info from the chunk's `ShapeEncoder` instance.
        byte_positions: (numpy.ndarray) Encoded byte positions from the chunk's `BytePositionsEncoder` instance.

    Returns:
        Length of the headers of chunk when serialized as int"""
    return len(version) + shape_info.nbytes + byte_positions.nbytes + HEADER_SIZE_BYTES


def infer_chunk_num_bytes(
    version: str,
    shape_info: np.ndarray,
    byte_positions: np.ndarray,
    data: Optional[Union[Sequence[bytes], Sequence[memoryview]]] = None,
    len_data: Optional[int] = None,
) -> int:
    """Calculates the number of bytes in a chunk without serializing it. Used by `LRUCache` to determine if a chunk can be cached.

    Args:
        version: (str) Version of hub library
        shape_info: (numpy.ndarray) Encoded shapes info from the chunk's `ShapeEncoder` instance.
        byte_positions: (numpy.ndarray) Encoded byte positions from the chunk's `BytePositionsEncoder` instance.
        data: (list) `_data` field of the chunk
        len_data: (int, optional) Number of bytes in the chunk

    Returns:
        Length of the chunk when serialized as int
    """
    # NOTE: Assumption: version string contains ascii characters only (ord(c) < 128)
    # NOTE: Assumption: len(version) < 256
    if len_data is None:
        len_data = sum(map(len, data))  # type: ignore

    header_size = infer_header_num_bytes(version, shape_info, byte_positions)
    return header_size + len_data


def serialize_chunk(
    version: str,
    shape_info: np.ndarray,
    byte_positions: np.ndarray,
    data: Union[Sequence[bytes], Sequence[memoryview]],
    len_data: Optional[int] = None,
) -> memoryview:
    """Serializes a chunk's headers and data into a single byte stream. This is how the chunk will be written to the storage provider.

    Args:
        version: (str) Version of hub library.
        shape_info: (numpy.ndarray) Encoded shapes info from the chunk's `ShapeEncoder` instance.
        byte_positions: (numpy.ndarray) Encoded byte positions from the chunk's `BytePositionsEncoder` instance.
        data: (list) `_data` field of the chunk.
        len_data: (int, optional) Number of bytes in the chunk.

    Returns:
        Serialized chunk as memoryview.
    """
    nbytes = infer_chunk_num_bytes(version, shape_info, byte_positions, data, len_data)
    flatbuff = bytearray(nbytes)
    offset = write_version(version, flatbuff)
    offset = write_shape_info(shape_info, flatbuff, offset)
    offset = write_byte_positions(byte_positions, flatbuff, offset)
    offset = write_actual_data(data, flatbuff, offset)
    return memoryview(flatbuff)


def write_version(version, buffer) -> int:
    """Writes version info to the buffer, returns offset."""
    len_version = len(version)
    buffer[0] = len_version
    buffer[1 : 1 + len_version] = version.encode("ascii")
    offset = 1 + len_version
    return offset


def write_shape_info(shape_info, buffer, offset) -> int:
    """Writes shape info to the buffer, takes offset into account and returns updated offset."""
    if shape_info.ndim == 1:
        offset += 8
    else:
        buffer[offset : offset + 8] = struct.pack("<ii", *shape_info.shape)
        offset += 8

        buffer[offset : offset + shape_info.nbytes] = shape_info.tobytes()
        offset += shape_info.nbytes
    return offset


def write_byte_positions(byte_positions, buffer, offset) -> int:
    """Writes byte positions info to the buffer, takes offset into account and returns updated offset."""
    if byte_positions.ndim == 1:
        offset += 4
    else:
        buffer[offset : offset + 4] = byte_positions.shape[0].to_bytes(4, "little")
        offset += 4

        buffer[offset : offset + byte_positions.nbytes] = byte_positions.tobytes()
        offset += byte_positions.nbytes
    return offset


def write_actual_data(data, buffer, offset) -> int:
    """Writes actual chunk data to the buffer, takes offset into account and returns updated offset"""
    for byts in data:
        n = len(byts)
        buffer[offset : offset + n] = byts
        offset += n
    return offset


def get_header_from_url(url: str):
    # Note: to be only used for chunks contains a single sample
    enc_dtype = np.dtype(hub.constants.ENCODING_DTYPE)
    itemsize = enc_dtype.itemsize

    headers = {"Range": "bytes=0-100"}

    request = Request(url, None, headers)
    byts = urlopen(request).read()

    len_version = byts[0]  # length of version string
    version = str(byts[1 : len_version + 1], "ascii")
    offset = 1 + len_version

    shape_info_nrows, shape_info_ncols = struct.unpack("<ii", byts[offset : offset + 8])
    shape_info_nbytes = shape_info_nrows * shape_info_ncols * itemsize
    offset += 8

    if shape_info_nbytes == 0:
        shape_info = np.array([], dtype=enc_dtype)
    else:
        shape_info = (
            np.frombuffer(byts[offset : offset + shape_info_nbytes], dtype=enc_dtype)
            .reshape(shape_info_nrows, shape_info_ncols)
            .copy()
        )
        offset += shape_info_nbytes

    byte_positions_rows = int.from_bytes(byts[offset : offset + 4], "little")
    byte_positions_nbytes = byte_positions_rows * 3 * itemsize
    offset += 4

    if byte_positions_nbytes == 0:
        byte_positions = np.array([], dtype=enc_dtype)
    else:
        byte_positions = (
            np.frombuffer(
                byts[offset : offset + byte_positions_nbytes], dtype=enc_dtype
            )
            .reshape(byte_positions_rows, 3)
            .copy()
        )
        offset += byte_positions_nbytes

    return version, shape_info, byte_positions, offset


def deserialize_chunk(
    byts: Union[bytes, memoryview], copy: bool = True, partial: bool = False
) -> Tuple[str, np.ndarray, np.ndarray, memoryview]:
    """Deserializes a chunk from the serialized byte stream. This is how the chunk can be accessed/modified after it is read from storage.

    Args:
        byts: (bytes) Serialized chunk.
        copy: (bool) If true, this function copies the byts while deserializing incase byts was a memoryview.
        partial: (bool) If true, the byts are only a part of the chunk.

    Returns:
        Tuple of:
        hub version used to create the chunk,
        encoded shapes info as numpy array,
        encoded byte positions as numpy array,
        chunk data as memoryview.

    Raises:
        IncompleteHeaderBytesError: For partial chunks, if the byts aren't complete to get the header.
    """
    incoming_mview = isinstance(byts, memoryview)
    byts = memoryview(byts)

    enc_dtype = np.dtype(hub.constants.ENCODING_DTYPE)
    itemsize = enc_dtype.itemsize

    # Read version
    len_version = byts[0]
    version = str(byts[1 : 1 + len_version], "ascii")
    offset = 1 + len_version

    # Read shape info
    shape_info_nrows, shape_info_ncols = struct.unpack("<ii", byts[offset : offset + 8])
    offset += 8
    shape_info_nbytes = shape_info_nrows * shape_info_ncols * itemsize
    if shape_info_nbytes == 0:
        shape_info = np.array([], dtype=enc_dtype)
    else:
        shape_info = (
            np.frombuffer(byts[offset : offset + shape_info_nbytes], dtype=enc_dtype)
            .reshape(shape_info_nrows, shape_info_ncols)
            .copy()
        )
        offset += shape_info_nbytes

    # Read byte positions
    byte_positions_rows = int.from_bytes(byts[offset : offset + 4], "little")
    offset += 4
    byte_positions_nbytes = byte_positions_rows * 3 * itemsize
    if byte_positions_nbytes == 0:
        byte_positions = np.array([], dtype=enc_dtype)
    else:
        byte_positions = (
            np.frombuffer(
                byts[offset : offset + byte_positions_nbytes], dtype=enc_dtype
            )
            .reshape(byte_positions_rows, 3)
            .copy()
        )
        offset += byte_positions_nbytes

    # Read data
    data = byts[offset:]
    if incoming_mview and copy:
        data = memoryview(bytes(data))
    return version, shape_info, byte_positions, data  # type: ignore


def serialize_chunkids(version: str, arr: np.ndarray) -> memoryview:
    """Serializes chunk ID encoders into a single byte stream. This is how the encoders will be written to the storage provider.

    Args:
        version: (str) Version of hub library.
        arr: (np.ndarray) Encoded chunk ids from a `ChunkIdEncoder` instance.

    Returns:
        Serialized chunk ids as memoryview.
    """
    len_version = len(version)
    flatbuff = bytearray(2 + len_version + arr.nbytes)

    # Write version
    len_version = len(version)
    flatbuff[0] = len_version
    flatbuff[1 : 1 + len_version] = version.encode("ascii")
    offset = 1 + len_version

    # write encoder dtype
    if version_compare(version, "2.7.6") >= 0:
        dtype = arr.dtype
        num_bytes = int(dtype.itemsize)
        flatbuff[offset] = num_bytes
        offset += 1

    # Write ids
    flatbuff[offset : offset + arr.nbytes] = arr.tobytes()
    offset += arr.nbytes
    return memoryview(flatbuff)


def deserialize_chunkids(
    byts: Union[bytes, memoryview]
) -> Tuple[str, np.ndarray, type]:
    """Deserializes a chunk ID encoder from the serialized byte stream. This is how the encoder can be accessed/modified after it is read from storage.

    Args:
        byts: (bytes) Serialized chunk ids.

    Returns:
        Tuple of: hub version used to create the chunk, encoded chunk ids as memoryview and dtype of the encoder.

    Raises:
        ValueError: If the bytes are not a valid chunk ID encoder.
    """
    byts = memoryview(byts)
    # Read version
    len_version = byts[0]
    version = str(byts[1 : 1 + len_version], "ascii")
    offset = 1 + len_version
    if version_compare(version, "2.7.6") < 0:
        # Read chunk ids
        ids = np.frombuffer(byts[offset:], dtype=np.uint32).reshape(-1, 2).copy()
        return version, ids, np.uint32
    else:
        # Read number of bytes per entry
        num_bytes = byts[offset]
        if num_bytes == 4:
            dtype = np.uint32
        elif num_bytes == 8:
            dtype = np.uint64  # type: ignore
        else:
            raise ValueError(f"Invalid number of bytes per entry: {num_bytes}")
        offset += 1
        # Read chunk ids
        ids = np.frombuffer(byts[offset:], dtype=dtype).reshape(-1, 2).copy()
        return version, ids, dtype


def serialize_sequence_or_creds_encoder(version: str, enc: np.ndarray) -> bytes:
    return len(version).to_bytes(1, "little") + version.encode("ascii") + enc.tobytes()


def deserialize_sequence_or_creds_encoder(
    byts: Union[bytes, memoryview], enc_type: str
) -> Tuple[str, np.ndarray]:
    dim = 2 if enc_type == "creds" else 3
    byts = memoryview(byts)
    len_version = byts[0]
    version = str(byts[1 : 1 + len_version], "ascii")
    enc = (
        np.frombuffer(byts[1 + len_version :], dtype=hub.constants.ENCODING_DTYPE)
        .reshape(-1, dim)
        .copy()
    )
    return version, enc


def check_sample_shape(shape, num_dims):
    if shape is not None and len(shape) != num_dims:
        raise TensorInvalidSampleShapeError(shape, num_dims)


def text_to_bytes(sample, dtype, htype):
    if isinstance(sample, hub.core.tensor.Tensor):
        try:
            if sample.htype == htype or sample.htype == "json" and htype == "list":
                return sample.tobytes(), sample.shape
        except (ValueError, NotImplementedError):  # sliced sample or tiled sample
            sample = sample.data()
    if htype in ("json", "list"):
        if isinstance(sample, np.ndarray):
            if htype == "list":
                sample = list(sample) if sample.dtype == object else sample.tolist()
            elif htype == "json":
                if sample.ndim == 0 or sample.dtype != object:
                    sample = sample.tolist()  # actually returns dict
                else:
                    sample = list(sample)
        validate_json_object(sample, dtype)
        byts = json.dumps(sample, cls=HubJsonEncoder).encode()
        shape = (len(sample),) if htype == "list" else (1,)
    else:  # htype == "text":
        if isinstance(sample, np.ndarray):
            sample = sample.tolist()
        if not isinstance(sample, str):
            raise TypeError("Expected str, received: " + str(sample))
        byts = sample.encode()
        shape = (1,)
    return byts, shape


def bytes_to_text(buffer, htype):
    buffer = bytes(buffer)
    if htype == "json":
        arr = np.empty(1, dtype=object)
        arr[0] = json.loads(bytes.decode(buffer), cls=HubJsonDecoder)
        return arr
    elif htype == "list":
        lst = json.loads(bytes.decode(buffer), cls=HubJsonDecoder)
        arr = np.empty(len(lst), dtype=object)
        arr[:] = lst
        return arr
    else:  # htype == "text":
        arr = np.array(bytes.decode(buffer)).reshape(
            1,
        )
    return arr


def serialize_text(
    incoming_sample: SampleValue,
    sample_compression: Optional[str],
    dtype: str,
    htype: str,
):
    """Converts the sample into bytes"""
    incoming_sample, shape = text_to_bytes(incoming_sample, dtype, htype)
    if sample_compression:
        incoming_sample = compress_bytes(incoming_sample, sample_compression)  # type: ignore
    return incoming_sample, shape


def serialize_numpy_and_base_types(
    incoming_sample: BaseTypes,
    sample_compression: Optional[str],
    chunk_compression: Optional[str],
    dtype: str,
    htype: str,
    min_chunk_size: int,
    break_into_tiles: bool = True,
    store_tiles: bool = False,
):
    """Converts the sample into bytes"""
    out = intelligent_cast(incoming_sample, dtype, htype)
    shape = out.shape
    tile_compression = chunk_compression or sample_compression

    if sample_compression is None:
        if out.nbytes > min_chunk_size and break_into_tiles:
            out = SampleTiles(out, tile_compression, min_chunk_size, store_tiles, htype)  # type: ignore
        else:
            out = out.tobytes()  # type: ignore
    else:
        ratio = get_compression_ratio(sample_compression)
        approx_compressed_size = out.nbytes * ratio

        if approx_compressed_size > min_chunk_size and break_into_tiles:
            out = SampleTiles(out, tile_compression, min_chunk_size, store_tiles, htype)  # type: ignore
        else:
            compressed_bytes = compress_array(out, sample_compression)
            out = compressed_bytes  # type: ignore

    return out, shape


def serialize_partial_sample_object(
    incoming_sample: PartialSample,
    sample_compression: Optional[str],
    chunk_compression: Optional[str],
    dtype: str,
    htype: str,
    min_chunk_size: int,
):
    shape = incoming_sample.shape
    return (
        SampleTiles(
            compression=sample_compression or chunk_compression,
            chunk_size=min_chunk_size,
            htype=htype,
            dtype=dtype,
            sample_shape=shape,
            tile_shape=incoming_sample.tile_shape,
        ),
        shape,
    )


def serialize_text_sample_object(
    incoming_sample: Sample, sample_compression: Optional[str]
):
    shape = incoming_sample.shape
    out = incoming_sample
    result = out.compressed_bytes(sample_compression)
    return result, shape


def serialize_sample_object(
    incoming_sample: Sample,
    sample_compression: Optional[str],
    chunk_compression: Optional[str],
    dtype: str,
    htype: str,
    min_chunk_size: int,
    break_into_tiles: bool = True,
    store_tiles: bool = False,
):
    shape = incoming_sample.shape
    tile_compression = chunk_compression or sample_compression

    out = incoming_sample
    if sample_compression:
        compression_type = get_compression_type(sample_compression)
        is_byte_compression = compression_type == BYTE_COMPRESSION
        if is_byte_compression and out.dtype != dtype:
            # Byte compressions don't store dtype, need to cast to expected dtype
            arr = intelligent_cast(out.array, dtype, htype)
            out = Sample(array=arr)

        compressed_bytes = out.compressed_bytes(sample_compression)

        if (
            compression_type not in (VIDEO_COMPRESSION, AUDIO_COMPRESSION)
            and len(compressed_bytes) > min_chunk_size
            and break_into_tiles
        ):
            out = SampleTiles(  # type: ignore
                out.array, tile_compression, min_chunk_size, store_tiles, htype
            )
        else:
            out = compressed_bytes  # type: ignore
    else:
        out = intelligent_cast(out.array, dtype, htype)  # type: ignore

        if out.nbytes > min_chunk_size and break_into_tiles:  # type: ignore
            out = SampleTiles(out, tile_compression, min_chunk_size, store_tiles, htype)  # type: ignore
        else:
            out = out.tobytes()  # type: ignore
    return out, shape


def serialize_tensor(
    incoming_sample: "hub.core.tensor.Tensor",
    sample_compression: Optional[str],
    chunk_compression: Optional[str],
    dtype: str,
    htype: str,
    min_chunk_size: int,
    break_into_tiles: bool = True,
    store_tiles: bool = False,
):
    def _return_numpy():
        return serialize_numpy_and_base_types(
            incoming_sample.numpy(),
            sample_compression,
            chunk_compression,
            dtype,
            htype,
            min_chunk_size,
            break_into_tiles,
            store_tiles,
        )

    if incoming_sample.meta.chunk_compression or chunk_compression:
        return _return_numpy()
    elif incoming_sample.meta.sample_compression == sample_compression:
        # Pass through
        try:
            return incoming_sample.tobytes(), incoming_sample.shape  # type: ignore
        except (
            ValueError,
            NotImplementedError,
        ) as e:  # Slice of sample or tiled sample
            return _return_numpy()
    else:
        return _return_numpy()
