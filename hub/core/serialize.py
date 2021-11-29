from hub.compression import BYTE_COMPRESSION, get_compression_type
from hub.core.tiling.sample_tiles import SampleTiles  # type: ignore
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

BaseTypes = Union[np.ndarray, list, int, float, bool, np.integer, np.floating, np.bool_]


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
    return len(version) + shape_info.nbytes + byte_positions.nbytes + len_data + 13  # type: ignore


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


def deserialize_chunk(
    byts: Union[bytes, memoryview], copy: bool = True
) -> Tuple[str, np.ndarray, np.ndarray, memoryview]:
    """Deserializes a chunk from the serialized byte stream. This is how the chunk can be accessed/modified after it is read from storage.

    Args:
        byts: (bytes) Serialized chunk.
        copy: (bool) If true, this function copies the byts while deserializing incase byts was a memoryview.

    Returns:
        Tuple of:
        hub version used to create the chunk,
        encoded shapes info as numpy array,
        encoded byte positions as numpy array,
        chunk data as memoryview.
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


def serialize_chunkids(version: str, ids: Sequence[np.ndarray]) -> memoryview:
    """Serializes chunk ID encoders into a single byte stream. This is how the encoders will be written to the storage provider.

    Args:
        version: (str) Version of hub library.
        ids: (list) Encoded chunk ids from a `ChunkIdEncoder` instance.

    Returns:
        Serialized chunk ids as memoryview.
    """
    len_version = len(version)
    flatbuff = bytearray(1 + len_version + sum([x.nbytes for x in ids]))

    # Write version
    len_version = len(version)
    flatbuff[0] = len_version
    flatbuff[1 : 1 + len_version] = version.encode("ascii")
    offset = 1 + len_version

    # Write ids
    for arr in ids:
        flatbuff[offset : offset + arr.nbytes] = arr.tobytes()
        offset += arr.nbytes

    return memoryview(flatbuff)


def deserialize_chunkids(byts: Union[bytes, memoryview]) -> Tuple[str, np.ndarray]:
    """Deserializes a chunk ID encoder from the serialized byte stream. This is how the encoder can be accessed/modified after it is read from storage.

    Args:
        byts: (bytes) Serialized chunk ids.

    Returns:
        Tuple of:
        hub version used to create the chunk,
        encoded chunk ids as memoryview.
    """
    byts = memoryview(byts)
    enc_dtype = np.dtype(hub.constants.ENCODING_DTYPE)

    # Read version
    len_version = byts[0]
    version = str(byts[1 : 1 + len_version], "ascii")
    offset = 1 + len_version

    # Read chunk ids
    ids = np.frombuffer(byts[offset:], dtype=enc_dtype).reshape(-1, 2).copy()
    return version, ids


def check_sample_shape(shape, num_dims):
    if len(shape) != num_dims:
        raise TensorInvalidSampleShapeError(shape, num_dims)


def text_to_bytes(sample, dtype, htype):
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
        incoming_sample = compress_bytes(incoming_sample, sample_compression)
    return incoming_sample, shape


def serialize_numpy_and_base_types(
    incoming_sample: BaseTypes,
    sample_compression: Optional[str],
    chunk_compression: Optional[str],
    dtype: str,
    htype: str,
    min_chunk_size: int,
    break_into_tiles: bool = True,
    store_uncompressed_tiles: bool = False,
):
    """Converts the sample into bytes"""
    incoming_sample = intelligent_cast(incoming_sample, dtype, htype)
    shape = incoming_sample.shape
    tile_compression = chunk_compression or sample_compression

    if sample_compression is None:
        if incoming_sample.nbytes > min_chunk_size and break_into_tiles:
            serialized_sample = SampleTiles(
                incoming_sample,
                tile_compression,
                min_chunk_size,
                store_uncompressed_tiles,
            )
        else:
            serialized_sample = incoming_sample.tobytes()
    else:
        compressed_bytes = compress_array(incoming_sample, sample_compression)
        if len(compressed_bytes) > min_chunk_size and break_into_tiles:
            serialized_sample = SampleTiles(
                incoming_sample,
                tile_compression,
                min_chunk_size,
                store_uncompressed_tiles,
            )
        else:
            serialized_sample = compressed_bytes
    return serialized_sample, shape


def serialize_sample_object(
    incoming_sample: Sample,
    sample_compression: Optional[str],
    chunk_compression: Optional[str],
    dtype: str,
    htype: str,
    min_chunk_size: int,
    break_into_tiles: bool = True,
    store_uncompressed_tiles: bool = False,
):
    is_byte_compression = get_compression_type(sample_compression) == BYTE_COMPRESSION
    shape = incoming_sample.shape
    tile_compression = chunk_compression or sample_compression
    if sample_compression:
        if is_byte_compression and incoming_sample.dtype != dtype:
            # Byte compressions don't store dtype, need to cast to expected dtype
            arr = intelligent_cast(incoming_sample.array, dtype, htype)
            incoming_sample = Sample(array=arr)
        compressed_bytes = incoming_sample.compressed_bytes(sample_compression)
        if len(compressed_bytes) > min_chunk_size and break_into_tiles:
            incoming_sample = SampleTiles(
                incoming_sample.array,
                tile_compression,
                min_chunk_size,
                store_uncompressed_tiles,
            )
        else:
            incoming_sample = compressed_bytes
    else:
        incoming_sample = intelligent_cast(incoming_sample.array, dtype, htype)
        if incoming_sample.nbytes > min_chunk_size and break_into_tiles:
            incoming_sample = SampleTiles(
                incoming_sample,
                tile_compression,
                min_chunk_size,
                store_uncompressed_tiles,
            )
        else:
            incoming_sample = incoming_sample.tobytes()
    return incoming_sample, shape
