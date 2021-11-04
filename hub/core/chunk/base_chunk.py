from abc import abstractmethod
from typing import List, Optional, Tuple
import numpy as np

import hub
from hub.compression import BYTE_COMPRESSION, IMAGE_COMPRESSIONS
from hub.core.fast_forwarding import ffw_chunk
from hub.core.meta.encode.byte_positions import BytePositionsEncoder
from hub.core.meta.encode.shape import ShapeEncoder
from hub.core.meta.tensor_meta import TensorMeta
from hub.core.serialize import deserialize_chunk, infer_chunk_num_bytes, serialize_chunk
from hub.core.storage.cachable import Cachable
import warnings


class BaseChunk(Cachable):
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
        self.data_bytes = data or bytearray()
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.tensor_meta = tensor_meta
        self.num_dims = (
            len(tensor_meta.max_shape) if self.tensor_meta.max_shape else None
        )
        self.is_text_like = self.htype in {"json", "list", "text"}
        self.compression = compression
        self.is_byte_compression = (
            hub.compression.get_compression_type(self.compression) == BYTE_COMPRESSION
        )
        self.uncompressed_samples = []

        self.version = hub.__version__

        self.shapes_encoder = ShapeEncoder(encoded_shapes)
        self.byte_positions_encoder = BytePositionsEncoder(encoded_byte_positions)
        self.is_convert_candidate = (
            self.htype == "image"
        ) or compression in IMAGE_COMPRESSIONS

        # These caches are only used for ChunkCompressed chunk.
        self._decompressed_samples: Optional[List[np.ndarray]] = None
        self._decompressed_bytes: Optional[memoryview] = None

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
            len_data=len(self.data_bytes),
        )

    def tobytes(self) -> memoryview:
        return serialize_chunk(
            self.version,
            self.shapes_encoder.array,
            self.byte_positions_encoder.array,
            [self.data_bytes],
        )

    @classmethod
    def frombuffer(cls, buffer: bytes, chunk_args: list, copy=True):
        if not buffer:
            return cls(*chunk_args)
        version, shapes, byte_positions, data = deserialize_chunk(buffer, copy=copy)
        chunk = cls(*chunk_args, shapes, byte_positions, data=data)
        chunk.version = version
        return chunk

    @abstractmethod
    def extend_if_has_space(self, incoming_sample):
        pass

    @abstractmethod
    def read_sample(
        self,
        local_sample_index: int,
        cast: bool = True,
        copy: bool = False,
    ):
        pass

    # TODO
    # @abstractmethod
    # def read_all_samples(self):
    #     pass

    @abstractmethod
    def update_sample(
        self, local_sample_index: int, new_buffer: memoryview, new_shape: Tuple[int]
    ):
        pass

    @property
    def memoryview_data(self):
        if isinstance(self.data_bytes, memoryview):
            return self.data_bytes
        return memoryview(self.data_bytes)

    def _make_data_bytearray(self):
        """Copies `self.data_bytes` into a bytearray if it is a memoryview."""

        # `_data` will be a `memoryview` if `frombuffer` is called.
        if isinstance(self.data_bytes, memoryview):
            self.data_bytes = bytearray(self.data_bytes)

    def prepare_for_write(self):
        ffw_chunk(self)
        self._make_data_bytearray()

    def register_sample_to_headers(
        self, incoming_num_bytes: Optional[int], sample_shape: tuple[int]
    ):
        """Registers a single sample to this chunk's header. A chunk should NOT exist without headers.

        Args:
            incoming_num_bytes (int): The length of the buffer that was used to
            sample_shape (Tuple[int]): Every sample that `num_samples` symbolizes is considered to have `sample_shape`.

        Raises:
            ValueError: If `incoming_num_bytes` is not divisible by `num_samples`.
        """

        self.shapes_encoder.register_samples(sample_shape, 1)
        if (
            incoming_num_bytes is not None
        ):  # incoming_num_bytes is not applicable for image compressions
            self.byte_positions_encoder.register_samples(incoming_num_bytes, 1)
        # self._clear_decompressed_caches()

    def convert_to_rgb(self, shape):
        if self.is_convert_candidate and hub.constants.CONVERT_GRAYSCALE:
            if self.num_dims is None:
                self.num_dims = len(shape)
            if len(shape) == 2 and self.num_dims == 3:
                warnings.warn(
                    "Grayscale images will be reshaped from (H, W) to (H, W, 1) to match tensor dimensions. This warning will be shown only once."
                )
                shape += (1,)  # type: ignore[assignment]
        return shape

    def can_fit_sample(self, sample_nbytes, buffer_nbytes=0):
        return self.num_data_bytes + buffer_nbytes + sample_nbytes < self.min_chunk_size

    def copy(self, chunk_args=None):
        return self.frombuffer(self.tobytes(), chunk_args)

    def update_meta_and_headers(self, sample_nbytes, shape):
        self.register_sample_to_headers(sample_nbytes, shape)
        self.tensor_meta.length += 1
        self.tensor_meta.update_shape_interval(shape)
