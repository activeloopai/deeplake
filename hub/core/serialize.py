from hub.core.meta.tensor_meta import TensorMeta
from hub.util.exceptions import TensorInvalidSampleShapeError
from hub.util.casting import intelligent_cast
from hub.util.json import HubJsonEncoder, validate_json_object
from hub.core.sample import Sample, SampleValue  # type: ignore
from hub.core.compression import compress_array, compress_bytes
from hub.client import config
from hub.compression import IMAGE_COMPRESSIONS
from typing import List, Optional, Sequence, Union, Tuple, Iterable
import hub
import numpy as np
import struct
import warnings
import json


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


def _serialize_input_sample(
    sample: SampleValue,
    sample_compression: Optional[str],
    expected_dtype: str,
    htype: str,
) -> Tuple[bytes, Tuple[int]]:
    """Converts the incoming sample into a buffer with the proper dtype and compression."""

    if htype in ("json", "list"):
        if isinstance(sample, np.ndarray):
            if htype == "list":
                if sample.dtype == object:
                    sample = list(sample)
                else:
                    sample = sample.tolist()
            elif htype == "json":
                if sample.ndim == 0:
                    sample = sample.tolist()  # actually returns dict
                elif sample.dtype == object:
                    sample = list(sample)
                else:
                    sample = sample.tolist()
        validate_json_object(sample, expected_dtype)
        byts = json.dumps(sample, cls=HubJsonEncoder).encode()
        if sample_compression:
            byts = compress_bytes(byts, compression=sample_compression)
        shape = (len(sample),) if htype == "list" else (1,)
        return byts, shape
    elif htype == "text":
        if isinstance(sample, np.ndarray):
            sample = sample.tolist()
        if not isinstance(sample, str):
            raise TypeError("Expected str, received: " + str(sample))
        byts = sample.encode()
        if sample_compression:
            byts = compress_bytes(byts, compression=sample_compression)
        return byts, (1,)

    if isinstance(sample, Sample):
        if (
            sample_compression
            and hub.compression.get_compression_type(sample_compression) == "byte"
        ):
            # Byte compressions don't store dtype info, so have to cast incoming samples to expected dtype
            arr = intelligent_cast(sample.array, expected_dtype, htype)
            sample = Sample(array=arr)
        buffer = sample.compressed_bytes(sample_compression)
        shape = sample.shape
    else:
        sample = intelligent_cast(sample, expected_dtype, htype)
        shape = sample.shape

        if sample_compression is not None:
            buffer = compress_array(sample, sample_compression)
        else:
            buffer = sample.tobytes()

    if len(shape) == 0:
        shape = (1,)

    return buffer, shape


def _check_input_samples_are_valid(
    num_bytes: List[int],
    shapes: List[Tuple[int]],
    min_chunk_size: int,
    sample_compression: Optional[str],
):
    """Iterates through all buffers/shapes and raises appropriate errors."""

    expected_dimensionality = None
    for nbytes, shape in zip(num_bytes, shapes):
        # check that all samples have the same dimensionality
        if expected_dimensionality is None:
            expected_dimensionality = len(shape)

        if nbytes > min_chunk_size:
            msg = f"Sorry, samples that exceed minimum chunk size ({min_chunk_size} bytes) are not supported yet (coming soon!). Got: {nbytes} bytes."
            if sample_compression is None:
                msg += "\nYour data is uncompressed, so setting `sample_compression` in `Dataset.create_tensor` could help here!"
            raise NotImplementedError(msg)

        if len(shape) != expected_dimensionality:
            raise TensorInvalidSampleShapeError(shape, expected_dimensionality)


def serialize_input_samples(
    samples: Union[Sequence[SampleValue], np.ndarray],
    meta: TensorMeta,
    min_chunk_size: int,
) -> Tuple[Union[memoryview, bytearray], List[int], List[Tuple[int]]]:
    """Casts, compresses, and serializes the incoming samples into a list of buffers and shapes.

    Args:
        samples (Union[Sequence[SampleValue], np.ndarray]): Ssequence of samples.
        meta (TensorMeta): Tensor meta. Will not be modified.
        min_chunk_size (int): Used to validate that all samples are appropriately sized.

    Raises:
        ValueError: Tensor meta should have it's dtype set.
        NotImplementedError: When extending tensors with Sample insatances.
        TypeError: When sample type is not understood.

    Returns:
        List[Tuple[memoryview, Tuple[int]]]: Buffers and their corresponding shapes for the input samples.
    """

    if meta.dtype is None:
        raise ValueError("Dtype must be set before input samples can be serialized.")

    sample_compression = meta.sample_compression
    chunk_compression = meta.chunk_compression
    dtype = meta.dtype
    htype = meta.htype

    if sample_compression or not hasattr(samples, "dtype"):
        buff = bytearray()
        nbytes = []
        shapes = []
        expected_dim = len(meta.max_shape)
        is_convert_candidate = (
            (htype == "image")
            or sample_compression in IMAGE_COMPRESSIONS
            or chunk_compression in IMAGE_COMPRESSIONS
        )

        for sample in samples:
            byts, shape = _serialize_input_sample(
                sample, sample_compression, dtype, htype
            )
            if (
                isinstance(sample, Sample)
                and is_convert_candidate
                and hub.constants.CONVERT_GRAYSCALE
            ):
                if not expected_dim:
                    expected_dim = len(shape)
                if len(shape) == 2 and expected_dim == 3:
                    warnings.warn(
                        f"Grayscale images will be reshaped from (H, W) to (H, W, 1) to match tensor dimensions. This warning will be shown only once."
                    )
                    shape += (1,)  # type: ignore[assignment]
            buff += byts
            nbytes.append(len(byts))
            shapes.append(shape)
    elif (
        isinstance(samples, np.ndarray)
        or np.isscalar(samples)
        or isinstance(samples, Sequence)
    ):
        samples = intelligent_cast(samples, dtype, htype)
        buff = memoryview(samples.tobytes())  # type: ignore
        if len(samples):
            shape = samples[0].shape
            nb = samples[0].nbytes
            if not shape:
                shape = (1,)
        else:
            shape = ()  # type: ignore
            nb = 0
        nbytes = [nb] * len(samples)
        shapes = [shape] * len(samples)
    elif isinstance(samples, Sample):
        # TODO
        raise NotImplementedError(
            "Extending with `Sample` instance is not supported yet."
        )
    else:
        raise TypeError(f"Cannot serialize samples of type {type(samples)}")
    _check_input_samples_are_valid(nbytes, shapes, min_chunk_size, sample_compression)
    return buff, nbytes, shapes
