from abc import abstractmethod
import struct
import numpy as np
from typing import List, Optional, Tuple, Union
import warnings

import hub
from hub.compression import (
    BYTE_COMPRESSION,
    IMAGE_COMPRESSION,
    VIDEO_COMPRESSION,
    get_compression_type,
)
from hub.constants import CONVERT_GRAYSCALE
from hub.core.fast_forwarding import ffw_chunk
from hub.core.linked_sample import LinkedSample
from hub.core.meta.encode.byte_positions import BytePositionsEncoder
from hub.core.meta.encode.shape import ShapeEncoder
from hub.core.meta.tensor_meta import TensorMeta
from hub.core.sample import Sample  # type: ignore
from hub.core.partial_sample import PartialSample
from hub.core.serialize import (
    deserialize_chunk,
    infer_chunk_num_bytes,
    serialize_chunk,
    serialize_numpy_and_base_types,
    serialize_sample_object,
    serialize_text,
    serialize_tensor,
    serialize_partial_sample_object,
    get_header_from_url,
)
from hub.core.storage.hub_memory_object import HubMemoryObject
from hub.core.tiling.sample_tiles import SampleTiles
from hub.util.exceptions import TensorInvalidSampleShapeError

InputSample = Union[
    Sample,
    np.ndarray,
    int,
    float,
    bool,
    dict,
    list,
    str,
    np.integer,
    np.floating,
    np.bool_,
]
SerializedOutput = Tuple[bytes, Tuple]


class BaseChunk(HubMemoryObject):
    def __init__(
        self,
        min_chunk_size: int,
        max_chunk_size: int,
        tensor_meta: TensorMeta,
        compression: Optional[str] = None,
        encoded_shapes: Optional[np.ndarray] = None,
        encoded_byte_positions: Optional[np.ndarray] = None,
        data: Optional[memoryview] = None,
    ):
        super().__init__()
        self._data_bytes: Union[bytearray, bytes, memoryview] = data or bytearray()
        self.version = hub.__version__
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

        self.tensor_meta = tensor_meta
        self.num_dims = len(tensor_meta.max_shape) if tensor_meta.max_shape else None
        self.is_text_like = (
            self.htype in {"json", "list", "text"} or self.tensor_meta.is_link
        )

        self.compression = compression
        compression_type = get_compression_type(compression)
        self.is_byte_compression = compression_type == BYTE_COMPRESSION
        self.is_image_compression = compression_type == IMAGE_COMPRESSION
        self.is_video_compression = compression_type == VIDEO_COMPRESSION
        self.is_convert_candidate = self.htype == "image" or self.is_image_compression

        self.shapes_encoder = ShapeEncoder(encoded_shapes)
        self.byte_positions_encoder = BytePositionsEncoder(encoded_byte_positions)

        if self.is_text_like and self.is_image_compression:
            raise ValueError("Can't use image compression with text data.")

        # These caches are only used for ChunkCompressed chunk.
        self.decompressed_samples: Optional[List[np.ndarray]] = None
        self.decompressed_bytes: Optional[bytes] = None

        # Whether tensor meta length is updated by chunk. Used by chunk engine while replacing chunks.
        self._update_tensor_meta_length: bool = (
            True  # Note: tensor meta shape interval is updated regardless.
        )

    @property
    def data_bytes(self) -> Union[bytearray, bytes, memoryview]:
        return self._data_bytes

    @data_bytes.setter
    def data_bytes(self, value: Union[bytearray, bytes, memoryview]):
        self._data_bytes = value

    @property
    def num_data_bytes(self) -> int:
        return len(self.data_bytes)

    @property
    def dtype(self):
        return self.tensor_meta.dtype

    @property
    def htype(self):
        return self.tensor_meta.htype

    @property
    def nbytes(self):
        """Calculates the number of bytes `tobytes` will be without having to call `tobytes`. Used by `LRUCache` to determine if this chunk can be cached."""
        return infer_chunk_num_bytes(
            self.version,
            self.shapes_encoder.array,
            self.byte_positions_encoder.array,
            len_data=self.num_data_bytes,
        )

    @property
    def memoryview_data(self):
        if isinstance(self.data_bytes, memoryview):
            return self.data_bytes
        return memoryview(self.data_bytes)

    @property
    def is_empty(self):
        return (
            self.num_data_bytes == 0
            and len(self.shapes_encoder.array) == 0
            and len(self.byte_positions_encoder.array) == 0
        )

    def tobytes(self) -> memoryview:
        return serialize_chunk(
            self.version,
            self.shapes_encoder.array,
            self.byte_positions_encoder.array,
            [self.data_bytes],
        )

    @classmethod
    def frombuffer(cls, buffer: bytes, chunk_args: list, copy=True, url=False):  # type: ignore
        if not buffer:
            return cls(*chunk_args)
        if url:
            version, shapes, byte_positions, header_size = get_header_from_url(
                buffer.decode("utf-8")
            )
            data = memoryview(buffer + struct.pack("<i", header_size))
        else:
            version, shapes, byte_positions, data = deserialize_chunk(buffer, copy=copy)
        chunk = cls(*chunk_args, shapes, byte_positions, data=data)  # type: ignore
        chunk.version = version
        chunk.is_dirty = False
        return chunk

    @abstractmethod
    def extend_if_has_space(self, incoming_samples, update_meta: bool = True) -> float:
        """Extends the chunk with the incoming samples."""

    @abstractmethod
    def read_sample(
        self,
        local_index: int,
        cast: bool = True,
        copy: bool = False,
        decompress: bool = True,
    ):
        """Reads a sample from the chunk."""

    @abstractmethod
    def update_sample(self, local_index: int, new_sample: InputSample):
        """Updates a sample in the chunk."""

    def _make_data_bytearray(self):
        """Copies `self.data_bytes` into a bytearray if it is a memoryview."""
        # data_bytes will be a memoryview if frombuffer is called.
        if isinstance(self.data_bytes, memoryview):
            self.data_bytes = bytearray(self.data_bytes)

    def prepare_for_write(self):
        ffw_chunk(self)
        self._make_data_bytearray()
        self.is_dirty = True

    def register_sample_to_headers(
        self, incoming_num_bytes: Optional[int], sample_shape: Tuple[int]
    ):
        """Registers a single sample to this chunk's header. A chunk should NOT exist without headers.

        Args:
            incoming_num_bytes (int): The length of the buffer that was used to
            sample_shape (Tuple[int]): Every sample that `num_samples` symbolizes is considered to have `sample_shape`.

        Raises:
            ValueError: If `incoming_num_bytes` is not divisible by `num_samples`.
        """
        self.shapes_encoder.register_samples(sample_shape, 1)
        # incoming_num_bytes is not applicable for image compressions
        if incoming_num_bytes is not None:
            self.byte_positions_encoder.register_samples(incoming_num_bytes, 1)

    def serialize_sample(
        self,
        incoming_sample: InputSample,
        sample_compression: Optional[str] = None,
        chunk_compression: Optional[str] = None,
        break_into_tiles: bool = True,
        store_uncompressed_tiles: bool = False,
    ) -> SerializedOutput:
        """Converts the sample into bytes"""
        dt, ht, min_chunk_size = self.dtype, self.htype, self.min_chunk_size
        if self.is_text_like:
            if isinstance(incoming_sample, LinkedSample):
                incoming_sample = incoming_sample.path
            incoming_sample, shape = serialize_text(
                incoming_sample, sample_compression, dt, ht  # type: ignore
            )
        elif isinstance(incoming_sample, Sample):
            incoming_sample, shape = serialize_sample_object(  # type: ignore
                incoming_sample,
                sample_compression,
                chunk_compression,
                dt,
                ht,
                min_chunk_size,
                break_into_tiles,
                store_uncompressed_tiles,
            )
            shape = self.convert_to_rgb(shape)
        elif isinstance(incoming_sample, PartialSample):
            incoming_sample, shape = serialize_partial_sample_object(
                incoming_sample,
                sample_compression,
                chunk_compression,
                dt,
                ht,
                min_chunk_size,
            )
        elif isinstance(incoming_sample, hub.core.tensor.Tensor):
            incoming_sample, shape = serialize_tensor(
                incoming_sample,
                sample_compression,
                chunk_compression,
                dt,
                ht,
                min_chunk_size,
                break_into_tiles,
                store_uncompressed_tiles,
            )
        elif isinstance(
            incoming_sample,
            (np.ndarray, list, int, float, bool, np.integer, np.floating, np.bool_),
        ):
            incoming_sample, shape = serialize_numpy_and_base_types(
                incoming_sample,
                sample_compression,
                chunk_compression,
                dt,
                ht,
                min_chunk_size,
                break_into_tiles,
                store_uncompressed_tiles,
            )
        elif isinstance(incoming_sample, SampleTiles):
            shape = incoming_sample.sample_shape
        else:
            raise TypeError(f"Cannot serialize sample of type {type(incoming_sample)}")
        shape = self.normalize_shape(shape)
        return incoming_sample, shape  # type: ignore

    def convert_to_rgb(self, shape):
        if self.is_convert_candidate and CONVERT_GRAYSCALE:
            if self.num_dims is None:
                self.num_dims = len(shape)
            if len(shape) == 2 and self.num_dims == 3:
                message = "Grayscale images will be reshaped from (H, W) to (H, W, 1) to match tensor dimensions. This warning will be shown only once."
                warnings.warn(message)
                shape += (1,)  # type: ignore[assignment]
        return shape

    def can_fit_sample(self, sample_nbytes, buffer_nbytes=0):
        return (
            self.num_data_bytes + buffer_nbytes + sample_nbytes <= self.min_chunk_size
        )

    def copy(self, chunk_args=None):
        return self.frombuffer(self.tobytes(), chunk_args)

    def register_in_meta_and_headers(self, sample_nbytes: Optional[int], shape):
        """Registers a new sample in meta and headers"""
        self.register_sample_to_headers(sample_nbytes, shape)
        if self._update_tensor_meta_length:
            self.tensor_meta.update_length(1)
        self.tensor_meta.update_shape_interval(shape)

    def update_in_meta_and_headers(
        self, local_index: int, sample_nbytes: Optional[int], shape
    ):
        """Updates an existing sample in meta and headers"""
        if sample_nbytes is not None:
            self.byte_positions_encoder[local_index] = sample_nbytes
        self.shapes_encoder[local_index] = shape
        self.tensor_meta.update_shape_interval(shape)

    def check_shape_for_update(self, local_index: int, shape):
        """Checks if the shape being assigned at the new index is valid."""
        expected_dimensionality = len(self.shapes_encoder[local_index])
        if expected_dimensionality != len(shape):
            raise TensorInvalidSampleShapeError(shape, expected_dimensionality)

    def create_updated_data(self, local_index: int, old_data, new_sample_bytes: bytes):
        if not old_data or self.byte_positions_encoder.is_empty():  # tiled sample
            return new_sample_bytes
        old_start_byte, old_end_byte = self.byte_positions_encoder[local_index]
        left_data = old_data[:old_start_byte]  # type: ignore
        right_data = old_data[old_end_byte:]  # type: ignore

        # preallocate
        total_new_bytes = len(left_data) + len(new_sample_bytes) + len(right_data)
        new_data = bytearray(total_new_bytes)

        # copy old data and add new data
        new_start_byte = old_start_byte
        new_end_byte = old_start_byte + len(new_sample_bytes)
        new_data[:new_start_byte] = left_data
        new_data[new_start_byte:new_end_byte] = new_sample_bytes
        new_data[new_end_byte:] = right_data
        return new_data

    def normalize_shape(self, shape):
        if shape is not None and len(shape) == 0:
            shape = (1,)
        return shape

    def write_tile(self, sample: SampleTiles):
        data, tile_shape = sample.yield_tile()
        self.data_bytes = data
        self.register_sample_to_headers(None, tile_shape)
        if sample.is_first_write:
            self.tensor_meta.update_shape_interval(sample.sample_shape)  # type: ignore
            if self._update_tensor_meta_length:
                self.tensor_meta.update_length(1)

    def _pop_sample(self):
        self.prepare_for_write()
        self.data_bytes = self.data_bytes[: self.byte_positions_encoder[-1][0]]
        self.shapes_encoder._pop()
        self.byte_positions_encoder._pop()

    def _get_partial_sample_tile(self, as_bytes=False):
        if not self._data_bytes and len(self.shapes_encoder._encoded) > 0:
            shape = self.shapes_encoder._encoded[0][:-1]
            if len(shape) and np.all(shape):
                if as_bytes:
                    return b"0" * int(
                        np.prod(np.array(shape, dtype=np.uint64))
                        * np.dtype(self.dtype).itemsize
                    )
                return np.zeros(shape, dtype=self.dtype)
        return None
