from typing import List, Optional, Union
import numpy as np
from hub.core.compression import compress_bytes, decompress_array, decompress_bytes
from hub.core.sample import Sample
from hub.core.serialize import (
    check_sample_shape,
    bytes_to_text,
    check_sample_size,
    serialize_numpy_and_base_types,
    text_to_bytes,
)
from hub.util.casting import intelligent_cast
from .base_chunk import BaseChunk

SampleValue = Union[Sample, np.ndarray, int, float, bool, dict, list, str]
SerializedOutput = tuple[bytes, Optional[tuple]]


class SampleCompressedChunk(BaseChunk):
    def serialize_sample(self, incoming_sample: SampleValue) -> SerializedOutput:
        dt, ht = self.dtype, self.htype
        if self.is_text_like:
            incoming_sample, shape = text_to_bytes(incoming_sample, dt, ht)
            incoming_sample = compress_bytes(incoming_sample, self.compression)
        elif isinstance(incoming_sample, Sample):
            shape = incoming_sample.shape
            shape = self.convert_to_rgb(shape)
            if self.is_byte_compression:
                # Byte compressions don't store dtype, need to cast to expected dtype
                arr = intelligent_cast(incoming_sample.array, dt, ht)
                incoming_sample = Sample(array=arr)
            incoming_sample = incoming_sample.compressed_bytes(self.compression)
        else:  # np.ndarray, int, float, bool
            incoming_sample, shape = serialize_numpy_and_base_types(
                incoming_sample, dt, ht, self.compression
            )
        if shape is not None and len(shape) == 0:
            shape = (1,)
        return incoming_sample, shape

    def extend_if_has_space(
        self, incoming_samples: Union[List[Union[bytes, Sample, np.array]], np.array]
    ) -> int:
        self.prepare_for_write()
        num_samples = 0

        for i, incoming_sample in enumerate(incoming_samples):
            serialized_sample, shape = self.serialize_sample(incoming_sample)
            self.num_dims = self.num_dims or len(shape)
            sample_nbytes = len(serialized_sample)
            check_sample_shape(shape, self.num_dims)
            check_sample_size(sample_nbytes, self.min_chunk_size, self.compression)
            # optimization so that even if this sample doesn't fit, it isn't recompressed next time we try
            incoming_samples[i] = Sample(
                buffer=serialized_sample, compression=self.compression, shape=shape
            )
            if not self.can_fit_sample(sample_nbytes):
                break
            self.data_bytes += serialized_sample
            self.update_meta_and_headers(sample_nbytes, shape)
            num_samples += 1
        return num_samples

    def read_sample(
        self, local_sample_index: int, cast: bool = True, copy: bool = False
    ):
        sb, eb = self.byte_positions_encoder[local_sample_index]
        buffer = self.memoryview_data[sb:eb]
        shape = self.shapes_encoder[local_sample_index]
        if self.is_text_like:
            buffer = decompress_bytes(buffer, compression=self.compression)
            buffer = bytes(buffer)
            return bytes_to_text(buffer, self.htype)

        sample = decompress_array(
            buffer, shape, dtype=self.dtype, compression=self.compression
        )
        if cast and sample.dtype != self.dtype:
            sample = sample.astype(self.dtype)
        return sample

    def update_sample(
        self, local_sample_index: int, new_buffer: memoryview, new_shape: tuple[int]
    ):
        raise NotImplementedError
