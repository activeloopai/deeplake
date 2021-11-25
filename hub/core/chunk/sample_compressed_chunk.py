import numpy as np
from typing import List, Sequence, Union
from hub.core.compression import decompress_array, decompress_bytes
from hub.core.sample import Sample  # type: ignore
from hub.core.serialize import (
    check_sample_shape,
    bytes_to_text,
    check_sample_size,
)
from .base_chunk import BaseChunk, InputSample


class SampleCompressedChunk(BaseChunk):
    def extend_if_has_space(
        self, incoming_samples: Union[Sequence[InputSample], np.ndarray]
    ) -> int:
        self.prepare_for_write()
        num_samples = 0

        for i, incoming_sample in enumerate(incoming_samples):
            serialized_sample, shape = self.sample_to_bytes(
                incoming_sample, self.compression, self.is_byte_compression
            )
            self.num_dims = self.num_dims or len(shape)
            sample_nbytes = len(serialized_sample)
            check_sample_shape(shape, self.num_dims)
            check_sample_size(sample_nbytes, self.min_chunk_size, self.compression)
            if serialized_sample and isinstance(incoming_samples, List):
                # optimization so that even if this sample doesn't fit, it isn't recompressed next time we try
                incoming_samples[i] = Sample(
                    buffer=serialized_sample, compression=self.compression, shape=shape
                )
            if not self.can_fit_sample(sample_nbytes):
                break
            self.data_bytes += serialized_sample  # type: ignore
            self.register_in_meta_and_headers(sample_nbytes, shape)
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
        self,
        local_sample_index: int,
        new_sample: InputSample,
    ):
        self.prepare_for_write()
        serialized_sample, shape = self.sample_to_bytes(
            new_sample, self.compression, self.is_byte_compression
        )

        self.check_shape_for_update(local_sample_index, shape)
        new_nb = len(serialized_sample)
        old_data = self.data_bytes
        self.data_bytes = self.create_buffer_with_updated_data(
            local_sample_index, old_data, serialized_sample
        )

        # update encoders and meta
        self.update_in_meta_and_headers(local_sample_index, new_nb, shape)
