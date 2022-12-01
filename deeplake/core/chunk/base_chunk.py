from abc import abstractmethod
import struct
import numpy as np
from typing import List, Optional, Tuple, Union
import warnings

import deeplake
from deeplake.compression import (
    BYTE_COMPRESSION,
    IMAGE_COMPRESSION,
    VIDEO_COMPRESSION,
    get_compression_type,
)
from deeplake.constants import CONVERT_GRAYSCALE
from deeplake.core.fast_forwarding import ffw_chunk
from deeplake.core.linked_sample import LinkedSample
from deeplake.core.meta.encode.byte_positions import BytePositionsEncoder
from deeplake.core.meta.encode.shape import ShapeEncoder
from deeplake.core.meta.tensor_meta import TensorMeta
from deeplake.core.partial_reader import PartialReader
from deeplake.core.sample import Sample  # type: ignore
from deeplake.core.partial_sample import PartialSample
from deeplake.core.serialize import (
    deserialize_chunk,
    infer_chunk_num_bytes,
    infer_header_num_bytes,
    serialize_chunk,
    serialize_numpy_and_base_types,
    serialize_sample_object,
    serialize_text,
    serialize_tensor,
    serialize_partial_sample_object,
    get_header_from_url,
    serialize_text_sample_object,
    serialize_polygons,
)
from deeplake.core.storage.deeplake_memory_object import DeepLakeMemoryObject
from deeplake.core.tiling.sample_tiles import SampleTiles
from deeplake.util.exceptions import TensorInvalidSampleShapeError
from deeplake.core.polygon import Polygons
from functools import reduce
from operator import mul

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


class BaseChunk(DeepLakeMemoryObject):
    def __init__(
        self,
        min_chunk_size: int,
        max_chunk_size: int,
        tiling_threshold: int,
        tensor_meta: TensorMeta,
        compression: Optional[str] = None,
        encoded_shapes: Optional[np.ndarray] = None,
        encoded_byte_positions: Optional[np.ndarray] = None,
        data: Optional[Union[memoryview, PartialReader]] = None,
    ):
        super().__init__()
        self._data_bytes: Union[bytearray, bytes, memoryview, PartialReader] = (
            data or bytearray()
        )
        self.version = deeplake.__version__
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.tiling_threshold = tiling_threshold

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
        self._item_size = None
        self._sample_size = None
        self.write_initialization_done = False

    @property
    def is_fixed_shape(self):
        return (
            self.tensor_meta.min_shape == self.tensor_meta.max_shape
            and not self.is_text_like
        )

    @property
    def item_size(self):
        # should only be called if self.is_fixed_shape
        if self._item_size is None:
            if self.dtype is None:
                raise ValueError("Can't get item size as dtype is not set.")
            self._item_size = np.dtype(self.dtype).itemsize
        return self._item_size

    @property
    def sample_size(self):
        # should only be called if self.is_fixed_shape
        shape = self.tensor_meta.max_shape
        if self._sample_size is None:
            self._sample_size = self.item_size * reduce(mul, shape, 1)
        return self._sample_size

    def get_byte_positions(self, local_index):
        # should only be called if self.is_fixed_shape
        return local_index * self.sample_size, (local_index + 1) * self.sample_size

    @property
    def is_partially_read_chunk(self):
        return isinstance(self.data_bytes, PartialReader)

    @property
    def data_bytes(self) -> Union[bytearray, bytes, memoryview, PartialReader]:
        return self._data_bytes

    @data_bytes.setter
    def data_bytes(self, value: Union[bytearray, bytes, memoryview, PartialReader]):
        self._data_bytes = value

    @property
    def num_data_bytes(self) -> int:
        if isinstance(self.data_bytes, PartialReader):
            enc = self.byte_positions_encoder
            num_samples = enc.num_samples
            if num_samples == 0:
                return 0
            first_data_start_byte = enc[0][0]
            last_data_end_byte = enc[num_samples - 1][1]
            return last_data_end_byte - first_data_start_byte

        return len(self.data_bytes)

    @property
    def dtype(self):
        return self.tensor_meta.dtype

    @property
    def htype(self):
        return self.tensor_meta.htype

    @property
    def num_samples(self) -> int:
        if not self.shapes_encoder.is_empty():
            return self.shapes_encoder.num_samples
        else:
            return self.byte_positions_encoder.num_samples

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
    def header_bytes(self):
        return infer_header_num_bytes(
            self.version, self.shapes_encoder.array, self.byte_positions_encoder.array
        )

    @property
    def memoryview_data(self):
        if isinstance(self.data_bytes, (memoryview, PartialReader)):
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
        if isinstance(self.data_bytes, PartialReader):
            self._make_data_bytearray()

        assert isinstance(self.data_bytes, (memoryview, bytearray, bytes))
        return serialize_chunk(
            self.version,
            self.shapes_encoder.array,
            self.byte_positions_encoder.array,
            [self.data_bytes],
        )

    @classmethod
    def frombuffer(cls, buffer: bytes, chunk_args: list, copy=True, url=False, partial=False):  # type: ignore
        if not buffer:
            return cls(*chunk_args)
        if url:
            version, shapes, byte_positions, header_size = get_header_from_url(
                buffer.decode("utf-8")
            )
            data = memoryview(buffer + struct.pack("<i", header_size))
        else:
            version, shapes, byte_positions, data = deserialize_chunk(buffer, copy=copy)
            if partial:
                data = None
        chunk = cls(*chunk_args, shapes, byte_positions, data=data)  # type: ignore
        chunk.version = version
        chunk.is_dirty = False
        return chunk

    @abstractmethod
    def extend_if_has_space(
        self, incoming_samples, update_meta: bool = True, end: bool = True, **kwargs
    ) -> float:
        """Extends the chunk with the incoming samples."""

    @abstractmethod
    def read_sample(
        self,
        local_index: int,
        cast: bool = True,
        copy: bool = False,
        decompress: bool = True,
        is_tile: bool = False,
    ):
        """Reads a sample from the chunk."""

    @abstractmethod
    def update_sample(self, local_index: int, new_sample: InputSample):
        """Updates a sample in the chunk."""

    def _make_data_bytearray(self):
        """Copies `self.data_bytes` into a bytearray if it is a memoryview."""
        # data_bytes will be a memoryview if frombuffer is called.
        if isinstance(self.data_bytes, PartialReader):
            chunk_bytes = self.data_bytes.get_all_bytes()
            self.data_bytes = bytearray(chunk_bytes[self.header_bytes :])
        elif isinstance(self.data_bytes, memoryview):
            self.data_bytes = bytearray(self.data_bytes)

    def prepare_for_write(self):
        if not self.write_initialization_done:
            ffw_chunk(self)
            self.write_initialization_done = True
        self._make_data_bytearray()
        self.is_dirty = True

    def serialize_sample(
        self,
        incoming_sample: InputSample,
        sample_compression: Optional[str] = None,
        chunk_compression: Optional[str] = None,
        break_into_tiles: bool = True,
        store_uncompressed_tiles: bool = False,
    ) -> SerializedOutput:
        """Converts the sample into bytes"""
        dt, ht, min_chunk_size, tiling_threshold = (
            self.dtype,
            self.htype,
            self.min_chunk_size,
            self.tiling_threshold,
        )
        if tiling_threshold < 0:
            break_into_tiles = False

        if isinstance(incoming_sample, LinkedSample):
            if self.tensor_meta.is_link:
                incoming_sample = incoming_sample.path
            else:
                raise ValueError(
                    "deeplake.link() samples can only be appended to linked tensors. To create linked tensors, include link in htype during create_tensor, for example 'link[image]'."
                )

        if self.is_text_like:
            if isinstance(incoming_sample, LinkedSample):
                incoming_sample = incoming_sample.path
            if incoming_sample is None:
                htype = "text" if self.tensor_meta.is_link else self.htype
                empty_mapping = {"text": "", "list": [], "json": {}}
                incoming_sample = empty_mapping[htype]

            if isinstance(incoming_sample, Sample):
                if incoming_sample.is_text_like:
                    incoming_sample, shape = serialize_text_sample_object(  # type: ignore
                        incoming_sample, sample_compression
                    )
                else:
                    htype = "Linked" if self.tensor_meta.is_link else self.htype
                    raise TypeError(
                        f"Cannot append to {htype} tensor with Sample object"
                    )
            else:
                incoming_sample, shape = serialize_text(
                    incoming_sample, sample_compression, dt, ht  # type: ignore
                )
        elif incoming_sample is None:
            shape = (0,) * self.num_dims if self.num_dims else None
            incoming_sample = b""
        elif isinstance(incoming_sample, Sample):
            incoming_sample, shape = serialize_sample_object(  # type: ignore
                incoming_sample,
                sample_compression,
                chunk_compression,
                dt,
                ht,
                tiling_threshold,
                break_into_tiles,
                store_uncompressed_tiles,
            )
        elif isinstance(incoming_sample, PartialSample):
            incoming_sample, shape = serialize_partial_sample_object(
                incoming_sample,
                sample_compression,
                chunk_compression,
                dt,
                ht,
                min_chunk_size,
            )
        elif isinstance(incoming_sample, deeplake.core.tensor.Tensor):
            incoming_sample, shape = serialize_tensor(
                incoming_sample,
                sample_compression,
                chunk_compression,
                dt,
                ht,
                tiling_threshold,
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
                tiling_threshold,
                break_into_tiles,
                store_uncompressed_tiles,
            )
        elif isinstance(incoming_sample, SampleTiles):
            shape = incoming_sample.sample_shape
        elif isinstance(incoming_sample, Polygons):
            incoming_sample, shape = serialize_polygons(
                incoming_sample, sample_compression, dt
            )
        else:
            raise TypeError(f"Cannot serialize sample of type {type(incoming_sample)}")
        shape = self.convert_to_rgb(shape)
        shape = self.normalize_shape(shape)
        return incoming_sample, shape  # type: ignore

    def convert_to_rgb(self, shape):
        if shape is not None and self.is_convert_candidate and CONVERT_GRAYSCALE:
            if self.num_dims is None:
                self.num_dims = len(shape)
            if len(shape) == 2 and self.num_dims == 3:
                message = "Grayscale images will be reshaped from (H, W) to (H, W, 1) to match tensor dimensions. This warning will be shown only once."
                warnings.warn(message)
                shape += (1,)  # type: ignore[assignment]
        return shape

    def can_fit_sample(self, sample_nbytes, buffer_nbytes=0):
        if self.num_data_bytes == 0:
            if self.tiling_threshold < 0:  # tiling disabled
                return True
            else:
                return buffer_nbytes + sample_nbytes <= self.tiling_threshold
        else:
            return (
                self.num_data_bytes + buffer_nbytes + sample_nbytes
                <= self.min_chunk_size
            )

    def copy(self, chunk_args=None):
        return self.frombuffer(self.tobytes(), chunk_args)

    def register_sample_to_headers(
        self,
        incoming_num_bytes: Optional[int],
        sample_shape: Tuple[int],
        num_samples: int = 1,
    ):
        """Registers a single sample to this chunk's header. A chunk should NOT exist without headers.

        Args:
            incoming_num_bytes (int): The length of the buffer that was used to
            sample_shape (Tuple[int]): Every sample that `num_samples` symbolizes is considered to have `sample_shape`.
            num_samples (int): Number of incoming samples.

        Raises:
            ValueError: If `incoming_num_bytes` is not divisible by `num_samples`.
        """
        # incoming_num_bytes is not applicable for image compressions
        if incoming_num_bytes is not None:
            self.byte_positions_encoder.register_samples(
                incoming_num_bytes, num_samples
            )
        if sample_shape is not None:
            if self.shapes_encoder.is_empty():
                padding = self.byte_positions_encoder.num_samples - num_samples
                self._fill_empty_shapes(sample_shape, padding)
            self.shapes_encoder.register_samples(sample_shape, num_samples)

    def register_in_meta_and_headers(
        self,
        sample_nbytes: Optional[int],
        shape,
        update_tensor_meta: bool = True,
        num_samples: int = 1,
    ):
        """Registers a new sample in meta and headers

        Args:
           sample_nbytes (Optional[int]): Paramter shat shows the numbero of bytes
           shape (Any): Parameter that shows the shape of the added elements
           update_commit_diff (bool): Parameter that shows if we need to update the commit diffs
           update_tensor_meta (bool): Parameter that shows if it is need to update tensor metas, in case of rechunk we do not need to update meta as we do not add new elements
           num_samples (int): Number of incoming samples.
        """
        self.register_sample_to_headers(sample_nbytes, shape, num_samples)
        if update_tensor_meta:
            self.update_tensor_meta(shape, num_samples)

    def update_tensor_meta(self, shape, num_samples):
        if self._update_tensor_meta_length:
            self.tensor_meta.update_length(num_samples)
        if shape is not None:
            self.tensor_meta.update_shape_interval(shape)

    def update_in_meta_and_headers(
        self, local_index: int, sample_nbytes: Optional[int], shape
    ):
        """Updates an existing sample in meta and headers"""
        if sample_nbytes is not None:
            self.byte_positions_encoder[local_index] = sample_nbytes
        if shape is not None:
            if self.shapes_encoder.is_empty():
                num_samples = self.byte_positions_encoder.num_samples
                self._fill_empty_shapes(shape, num_samples)
            self.shapes_encoder[local_index] = shape
            self.tensor_meta.update_shape_interval(shape)

    def check_shape_for_update(self, shape):
        """Checks if the shape being assigned at the new index is valid."""
        if shape is None:
            return
        max_shape = self.tensor_meta.max_shape
        if max_shape:
            expected_dimensionality = len(max_shape)
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

    def pop_multiple(self, num_samples):
        self.prepare_for_write()

        if not self.byte_positions_encoder.is_empty():
            total_samples = self.shapes_encoder.num_samples
            starting_byte_first_popped_sample = self.byte_positions_encoder[
                total_samples - num_samples
            ][0]
            self.data_bytes = self.data_bytes[:starting_byte_first_popped_sample]

        for _ in range(num_samples):
            if not self.shapes_encoder.is_empty():
                self.shapes_encoder.pop()
            if not self.byte_positions_encoder.is_empty():
                self.byte_positions_encoder.pop()

    def _get_partial_sample_tile(self, as_bytes=False):
        if (
            not isinstance(self.data_bytes, PartialReader)
            and not self.data_bytes
            and len(self.shapes_encoder._encoded) > 0
        ):
            shape = self.shapes_encoder._encoded[0][:-1]
            if len(shape) and np.all(shape):
                if as_bytes:
                    return b"0" * int(
                        np.prod(np.array(shape, dtype=np.uint64))
                        * np.dtype(self.dtype).itemsize
                    )
                return np.zeros(shape, dtype=self.dtype)
        return None

    def pop(self, index):
        self.prepare_for_write()
        sb, eb = self.byte_positions_encoder[index]
        self.data_bytes = self.data_bytes[:sb] + self.data_bytes[eb:]
        if not self.shapes_encoder.is_empty():
            self.shapes_encoder.pop(index)
        if not self.byte_positions_encoder.is_empty():
            self.byte_positions_encoder.pop(index)

    def _fill_empty_shapes(self, shape, num_samples):
        dims = len(shape)
        self.num_dims = self.num_dims or dims
        if num_samples > 0:
            empty_shape = (0,) * dims
            self.shapes_encoder.register_samples(empty_shape, num_samples)
            self.tensor_meta.update_shape_interval(empty_shape)

    @property
    def is_empty_tensor(self):
        return len(self.tensor_meta.max_shape) == 0 and len(self.data_bytes) == 0

    def _text_sample_to_byte_string(self, sample):
        try:
            return sample.encode("utf-8")
        except AttributeError:
            try:
                return sample.tolist().encode("utf-8")
            except AttributeError:  # None
                return b""
