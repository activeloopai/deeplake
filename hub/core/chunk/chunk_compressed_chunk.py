from typing import List, Tuple, Union

import numpy as np
from hub.compression import BYTE_COMPRESSION, get_compression_type
from hub.core.chunk.uncompressed_chunk import SerializedOutput
from hub.core.compression import compress_bytes, decompress_bytes

from hub.core.sample import Sample
from hub.core.serialize import (
    bytes_to_text,
    serialize_numpy_and_base_types,
    text_to_bytes,
)
from .base_chunk import BaseChunk

SampleValue = Union[bytes, Sample, np.ndarray, int, float, bool, dict, list, str]


class ChunkCompressedChunk(BaseChunk):
    """Responsibility: Case where we are using chunk-wise compression.
    Case:
        - sample_compression=None
        - chunk_compression=compressed
    Input pipeline:
        - hub.read(...) ->
            - decompress
            - add to uncompressed data
            - re-compress all data together
            - check if newly re-compressed data fits in this chunk, if not then create new chunk
        - numpy -> compressed bytes
            - add to uncompressed data
            - re-compress all data together
            - check if newly re-compressed data fits in this chunk, if not then create new chunk
    """

    def serialize_sample(self, incoming_sample: SampleValue) -> SerializedOutput:
        dt, ht = self.dtype, self.htype
        if self.is_text_like:
            incoming_sample, shape = text_to_bytes(incoming_sample, dt, ht)
        elif isinstance(incoming_sample, Sample):
            shape = incoming_sample.shape
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
        for i, incoming_sample in enumerate(incoming_samples):
            serialized_sample, shape = self.serialize_sample(incoming_sample)
            self.uncompressed_samples.append(serialized_sample)

            # TODO: optimize this
            compressed_samples = compress_bytes(
                self.uncompressed_samples, self.compression
            )

            if len(compressed_samples) > self.max_chunk_size:
                self.uncompressed_samples = self.uncompressed_samples[:-1]
                return i
            self.samples = compressed_samples
            self.shapes.append(shape)

            if get_compression_type(self.compression) == BYTE_COMPRESSION:
                self.register_sample_to_headers(incoming_num_bytes=len(serialized_sample), sample_shape=shape)  # type: ignore
            else:
                # Byte positions are not relevant for chunk wise image compressions, so incoming_num_bytes=None.
                self.register_sample_to_headers(incoming_num_bytes=None, sample_shape=shape)  # type: ignore

        return len(incoming_samples)

    def read_sample(
        self, local_sample_index: int, cast: bool = True, copy: bool = False
    ):
        sb, eb = self.byte_positions_encoder[local_sample_index]
        shape = self.shapes_encoder[local_sample_index]
        if self.is_text_like:
            decompressed = self.decompressed_data(
                compression=self.compression
            )  # TODO: implement this
            buffer = decompressed[sb:eb]
            buffer = bytes(buffer)
            return bytes_to_text(buffer, self.htype)

        if get_compression_type(self.compression) == BYTE_COMPRESSION:
            decompressed = self.decompressed_data(
                compression=self.compression
            )  # TODO: implement this
            return np.frombuffer(decompressed[sb:eb], dtype=self.dtype).reshape(shape)
        else:
            return self.decompressed_samples()[
                local_sample_index
            ]  # TODO: implement this

    def update_sample(
        self, local_sample_index: int, new_buffer: memoryview, new_shape: Tuple[int]
    ):
        raise NotImplementedError
