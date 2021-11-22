from typing import List, Sequence, Union
from hub.core.serialize import (
    check_sample_shape,
    bytes_to_text,
    check_sample_size,
)
from hub.util.casting import intelligent_cast
from .base_chunk import BaseChunk, InputSample
import numpy as np


class UncompressedChunk(BaseChunk):
    def extend_if_has_space(
        self, incoming_samples: Union[Sequence[InputSample], np.ndarray]
    ) -> int:
        self.prepare_for_write()
        if isinstance(incoming_samples, np.ndarray):
            return self._extend_if_has_space_numpy(incoming_samples)
        return self._extend_if_has_space_sequence(incoming_samples)

    def _extend_if_has_space_numpy(self, incoming_samples: np.ndarray):
        num_samples = 0
        buffer_size = 0
        for incoming_sample in incoming_samples:
            shape = incoming_sample.shape
            shape = self.normalize_shape(shape)
            self.num_dims = self.num_dims or len(shape)
            sample_nbytes = incoming_sample.nbytes
            check_sample_shape(shape, self.num_dims)
            check_sample_size(sample_nbytes, self.min_chunk_size, self.compression)
            if not self.can_fit_sample(sample_nbytes, buffer_size):
                break
            buffer_size += sample_nbytes
            num_samples += 1
        samples = incoming_samples[:num_samples]
        samples = intelligent_cast(samples, self.dtype, self.htype)
        self.data_bytes += samples.tobytes()

        if num_samples > 0:
            shape = samples[0].shape if len(samples[0].shape) > 0 else (1,)
            sample_nbytes = samples[0].nbytes
            for _ in range(num_samples):
                self.register_in_meta_and_headers(sample_nbytes, shape)

        return num_samples

    def _extend_if_has_space_sequence(self, incoming_samples: Sequence[InputSample]):
        num_samples = 0
        for incoming_sample in incoming_samples:
            serialized_sample, shape = self.sample_to_bytes(
                incoming_sample, None, False
            )
            self.num_dims = self.num_dims or len(shape)
            sample_nbytes = len(serialized_sample)
            check_sample_shape(shape, self.num_dims)
            check_sample_size(sample_nbytes, self.min_chunk_size, self.compression)
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
            buffer = bytes(buffer)
            return bytes_to_text(buffer, self.htype)

        buffer = bytes(buffer) if copy else buffer
        return np.frombuffer(buffer, dtype=self.dtype).reshape(shape)

    def update_sample(
        self,
        local_sample_index: int,
        new_sample: InputSample,
    ):
        self.prepare_for_write()
        serialized_sample, shape = self.sample_to_bytes(new_sample, None, False)
        self.check_shape_for_update(local_sample_index, shape)
        new_nb = len(serialized_sample)

        old_data = self.data_bytes
        self.data_bytes = self.create_buffer_with_updated_data(
            local_sample_index, old_data, serialized_sample
        )

        # update encoders and meta
        self.update_in_meta_and_headers(local_sample_index, new_nb, shape)
