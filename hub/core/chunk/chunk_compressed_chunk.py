from typing import List, Optional, Tuple, Union
import numpy as np
from hub.core.chunk.uncompressed_chunk import SerializedOutput
from hub.core.compression import (
    compress_bytes,
    compress_multiple,
    decompress_bytes,
    decompress_multiple,
)
from hub.core.sample import Sample
from hub.core.serialize import (
    bytes_to_text,
    check_sample_shape,
    check_sample_size,
    serialize_numpy_and_base_types,
    text_to_bytes,
)
from hub.util.casting import intelligent_cast
from hub.util.exceptions import SampleDecompressionError
from .base_chunk import BaseChunk

SampleValue = Union[bytes, Sample, np.ndarray, int, float, bool, dict, list, str]


class ChunkCompressedChunk(BaseChunk):
    def serialize_sample(self, incoming_sample: SampleValue) -> SerializedOutput:
        dt, ht = self.dtype, self.htype
        if self.is_text_like:
            incoming_sample, shape = text_to_bytes(incoming_sample, dt, ht)
        elif isinstance(incoming_sample, Sample):
            shape = incoming_sample.shape
            shape = self.convert_to_rgb(shape)
            incoming_sample = incoming_sample.uncompressed_bytes
        elif isinstance(incoming_sample, bytes):
            shape = None
        else:  # np.ndarray, int, float, bool
            incoming_sample, shape = serialize_numpy_and_base_types(
                incoming_sample, dt, ht, None
            )
        if shape is not None and len(shape) == 0:
            shape = (1,)
        return incoming_sample, shape

    def extend_if_has_space(
        self, incoming_samples: Union[List[Union[bytes, Sample, np.array]], np.array]
    ) -> int:
        self.prepare_for_write()
        if self.is_byte_compression:
            return self.extend_if_has_space_byte_compression(incoming_samples)
        return self.extend_if_has_space_image_compression(incoming_samples)

    def extend_if_has_space_byte_compression(
        self, incoming_samples: Union[List[Union[bytes, Sample, np.array]], np.array]
    ):
        num_samples = 0
        buffer = (
            bytearray(self.decompressed_bytes(compression=self.compression))
            if self.data_bytes
            else bytearray()
        )
        for incoming_sample in incoming_samples:
            serialized_sample, shape = self.serialize_sample(incoming_sample)
            self.num_dims = self.num_dims or len(shape)
            sample_nbytes = len(serialized_sample)
            check_sample_shape(shape, self.num_dims)
            check_sample_size(sample_nbytes, self.min_chunk_size, self.compression)
            buffer += serialized_sample
            # TODO: optimize this
            compressed_bytes = compress_bytes(buffer, self.compression)

            if len(compressed_bytes) > self.min_chunk_size:
                self._decompressed_bytes = buffer[:-sample_nbytes]
                break

            self.data_bytes = compressed_bytes
            self.update_meta_and_headers(sample_nbytes, shape)
            num_samples += 1
            self._decompressed_bytes = buffer
        return num_samples

    def extend_if_has_space_image_compression(
        self, incoming_samples: Union[List[Union[Sample, np.array]], np.array]
    ):
        num_samples = 0
        buffer_list = self.decompressed_samples() if self.data_bytes else []
        for incoming_sample in incoming_samples:
            if isinstance(incoming_sample, bytes):
                raise ValueError(
                    "Chunkwise image compression is not applicable on bytes."
                )
            incoming_sample = intelligent_cast(incoming_sample, self.dtype, self.htype)
            if isinstance(incoming_sample, Sample):
                incoming_sample = incoming_sample.array
            shape = incoming_sample.shape
            if len(shape) == 0:
                shape = (1,)
            self.num_dims = self.num_dims or len(shape)
            check_sample_shape(shape, self.num_dims)
            buffer_list.append(incoming_sample)

            # TODO: optimize this
            compressed_bytes = compress_multiple(buffer_list, self.compression)

            if len(compressed_bytes) > self.min_chunk_size:
                break

            self.data_bytes = compressed_bytes
            # Byte positions are not relevant for chunk wise image compressions, so incoming_num_bytes=None.
            self.update_meta_and_headers(None, shape)
            num_samples += 1
            self._decompressed_samples = buffer_list

        return num_samples

    def decompressed_samples(
        self,
        compression: Optional[str] = None,
        dtype: Optional[Union[np.dtype, str]] = None,
    ) -> List[np.ndarray]:
        """Applicable only for compressed chunks. Returns samples contained in this chunk as a list of numpy arrays."""
        if not self._decompressed_samples:
            shapes = [
                self.shapes_encoder[i] for i in range(self.shapes_encoder.num_samples)
            ]
            self._decompressed_samples = decompress_multiple(
                self.data_bytes, shapes, dtype, compression
            )
        return self._decompressed_samples

    def decompressed_bytes(self, compression: str) -> memoryview:
        """Applicable only for chunks compressed using a byte compression. Returns the contents of the chunk as a decompressed buffer."""
        if self._decompressed_bytes is None:
            try:
                self._decompressed_bytes = decompress_bytes(
                    self.data_bytes, compression
                )
            except SampleDecompressionError:
                raise ValueError(
                    "Chunk.decompressed_bytes() can not be called on chunks compressed with image compressions. Use Chunk.get_samples() instead."
                )
        return self._decompressed_bytes

    # def _clear_decompressed_caches(self):
    #     self._decompressed_samples = None
    #     self._decompressed_bytes = None

    def read_sample(
        self, local_sample_index: int, cast: bool = True, copy: bool = False
    ):
        if not self.is_byte_compression:
            return self.decompressed_samples()[local_sample_index]

        sb, eb = self.byte_positions_encoder[local_sample_index]
        shape = self.shapes_encoder[local_sample_index]
        decompressed = memoryview(self.decompressed_bytes(compression=self.compression))
        buffer = decompressed[sb:eb]
        if self.is_text_like:
            buffer = bytes(buffer)
            return bytes_to_text(buffer, self.htype)
        return np.frombuffer(decompressed[sb:eb], dtype=self.dtype).reshape(shape)

    def update_sample(
        self, local_sample_index: int, new_buffer: memoryview, new_shape: Tuple[int]
    ):
        raise NotImplementedError
