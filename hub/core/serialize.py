from hub.core.meta.tensor_meta import TensorMeta
from hub.util.exceptions import TensorInvalidSampleShapeError
from hub.util.casting import intelligent_cast
from hub.core.sample import Sample, SampleValue  # type: ignore
from hub.core.compression import compress_array
from typing import List, Optional, Sequence, Union, Tuple

import hub
import numpy as np


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
    return len(version) + shape_info.nbytes + byte_positions.nbytes + len_data + 13


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
    flatbuff = np.zeros(nbytes, dtype=np.byte)

    # Write version
    len_version = len(version)
    flatbuff[0] = len_version
    flatbuff[1 : 1 + len_version] = list(map(ord, version))
    offset = 1 + len_version

    # Write shape info
    if shape_info.ndim == 1:
        flatbuff[offset : offset + 8] = np.zeros(8, dtype=np.byte)
        offset += 8
    else:
        flatbuff[offset : offset + 8] = np.array(shape_info.shape, dtype=np.int32).view(
            np.byte
        )
        offset += 8
        flatbuff[offset : offset + shape_info.nbytes] = shape_info.reshape(-1).view(
            np.byte
        )
        offset += shape_info.nbytes

    # Write byte positions
    if byte_positions.ndim == 1:
        flatbuff[offset : offset + 4] = np.zeros(4, dtype=np.byte)
        offset += 4
    else:
        flatbuff[offset : offset + 4] = np.int32(byte_positions.shape[0]).view(
            (np.byte, 4)
        )
        offset += 4
        flatbuff[offset : offset + byte_positions.nbytes] = byte_positions.reshape(
            -1
        ).view(np.byte)
        offset += byte_positions.nbytes

    # Write actual data
    for byts in data:
        n = len(byts)
        flatbuff[offset : offset + n] = np.frombuffer(byts, dtype=np.byte)
        offset += n
    return memoryview(flatbuff.tobytes())


def deserialize_chunk(
    byts: Union[bytes, memoryview]
) -> Tuple[str, np.ndarray, np.ndarray, memoryview]:
    """Deserializes a chunk from the serialized byte stream. This is how the chunk can be accessed/modified after it is read from storage.

    Args:
        byts: (bytes) Serialized chunk.

    Returns:
        Tuple of:
        hub version used to create the chunk,
        encoded shapes info as numpy array,
        encoded byte positions as numpy array,
        chunk data as memoryview.
    """
    enc_dtype = np.dtype(hub.constants.ENCODING_DTYPE)

    buff = np.frombuffer(byts, dtype=np.byte)

    # Read version
    len_version = buff[0]
    version = "".join(map(chr, buff[1 : 1 + len_version]))
    offset = 1 + len_version

    # Read shape info
    shape_info_shape = buff[offset : offset + 8].view(np.int32)
    offset += 8
    shape_info_nbytes = np.prod(shape_info_shape) * enc_dtype.itemsize
    if shape_info_nbytes == 0:
        shape_info = np.array([], dtype=enc_dtype)
    else:
        shape_info = (
            buff[offset : offset + shape_info_nbytes]
            .view(enc_dtype)
            .reshape(shape_info_shape)
            .copy()
        )
        offset += shape_info_nbytes

    # Read byte positions
    byte_positions_rows = buff[offset : offset + 4].view(np.int32)[0]
    offset += 4
    byte_positions_nbytes = byte_positions_rows * 3 * enc_dtype.itemsize
    if byte_positions_nbytes == 0:
        byte_positions = np.array([], dtype=enc_dtype)
    else:
        byte_positions = (
            buff[offset : offset + byte_positions_nbytes]
            .view(enc_dtype)
            .reshape(byte_positions_rows, 3)
            .copy()
        )
        offset += byte_positions_nbytes

    # Read data
    data = memoryview(buff[offset:].tobytes())

    return version, shape_info, byte_positions, data


def serialize_chunkids(version: str, ids: Sequence[np.ndarray]) -> memoryview:
    """Serializes chunk ID encoders into a single byte stream. This is how the encoders will be written to the storage provider.

    Args:
        version: (str) Version of hub library.
        ids: (list) Encoded chunk ids from a `ChunkIdEncoder` instance.

    Returns:
        Serialized chunk ids as memoryview.
    """
    len_version = len(version)
    flatbuff = np.zeros(1 + len_version + sum([x.nbytes for x in ids]), dtype=np.byte)

    # Write version
    len_version = len(version)
    flatbuff[0] = len_version
    flatbuff[1 : 1 + len_version] = list(map(ord, version))
    offset = 1 + len_version

    # Write ids
    for arr in ids:
        flatbuff[offset : offset + arr.nbytes] = arr.view(np.byte).reshape(-1)
        offset += arr.nbytes

    return memoryview(flatbuff.tobytes())


def deserialize_chunkids(byts: Union[bytes, memoryview]) -> Tuple[str, np.ndarray]:
    """Deserializes a chunk ID encoder from the serialized byte stream. This is how the encoder can be accessed/modified after it is read from storage.

    Args:
        byts: (bytes) Serialized chunk ids.

    Returns:
        Tuple of:
        hub version used to create the chunk,
        encoded chunk ids as memoryview.
    """
    enc_dtype = np.dtype(hub.constants.ENCODING_DTYPE)

    buff = np.frombuffer(byts, dtype=np.byte)

    # Read version
    len_version = buff[0]
    version = "".join(map(chr, buff[1 : 1 + len_version]))
    offset = 1 + len_version

    # Read chunk ids
    ids = buff[offset:].view(enc_dtype).reshape(-1, 2).copy()

    return version, ids


def _serialize_input_sample(
    sample: SampleValue,
    sample_compression: Optional[str],
    expected_dtype: np.dtype,
    htype: str,
) -> Tuple[bytes, Tuple[int]]:
    """Converts the incoming sample into a buffer with the proper dtype and compression."""

    if isinstance(sample, Sample):
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
    buffer_and_shapes: List, min_chunk_size: int, sample_compression: Optional[str]
):
    """Iterates through all buffers/shapes and raises appropriate errors."""

    expected_dimensionality = None
    for buffer, shape in buffer_and_shapes:
        # check that all samples have the same dimensionality
        if expected_dimensionality is None:
            expected_dimensionality = len(shape)

        if len(buffer) > min_chunk_size:
            msg = f"Sorry, samples that exceed minimum chunk size ({min_chunk_size} bytes) are not supported yet (coming soon!). Got: {len(buffer)} bytes."
            if sample_compression is None:
                msg += "\nYour data is uncompressed, so setting `sample_compression` in `Dataset.create_tensor` could help here!"
            raise NotImplementedError(msg)

        if len(shape) != expected_dimensionality:
            raise TensorInvalidSampleShapeError(shape, expected_dimensionality)


def serialize_input_samples(
    samples: Union[Sequence[SampleValue], SampleValue],
    meta: TensorMeta,
    min_chunk_size: int,
) -> List[Tuple[memoryview, Tuple[int]]]:
    """Casts, compresses, and serializes the incoming samples into a list of buffers and shapes.

    Args:
        samples (Union[Sequence[SampleValue], SampleValue]): Either a single sample or sequence of samples.
        meta (TensorMeta): Tensor meta. Will not be modified.
        min_chunk_size (int): Used to validate that all samples are appropriately sized.

    Raises:
        ValueError: Tensor meta should have it's dtype set.

    Returns:
        List[Tuple[memoryview, Tuple[int]]]: Buffers and their corresponding shapes for the input samples.
    """

    if meta.dtype is None:
        raise ValueError("Dtype must be set before input samples can be serialized.")

    sample_compression = meta.sample_compression
    dtype = np.dtype(meta.dtype)
    htype = meta.htype

    serialized = []
    for sample in samples:
        byts, shape = _serialize_input_sample(sample, sample_compression, dtype, htype)
        buffer = memoryview(byts)
        serialized.append((buffer, shape))

    _check_input_samples_are_valid(serialized, min_chunk_size, dtype)
    return serialized
