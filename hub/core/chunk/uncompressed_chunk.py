from typing import List, Optional, Tuple, Union

from numpy.core.numerictypes import nbytes
from hub.core.sample import Sample
from hub.core.serialize import (
    check_sample_shape,
    bytes_to_text,
    check_sample_size,
    serialize_numpy_and_base_types,
    text_to_bytes,
)
from .base_chunk import BaseChunk
import numpy as np

SampleValue = Union[bytes, Sample, np.ndarray, int, float, bool, dict, list, str]
SerializedOutput = tuple[bytes, Optional[tuple]]


class UncompressedChunk(BaseChunk):
    """Responsibility: Case where we aren't using any compression.
    Case:
        - sample_compression=None
        - chunk_compression=None
    Input pipeline:
        - hub.read(...) -> numpy
        - numpy -> numpy
    """

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
                incoming_sample, dt, ht, self.compression
            )
        if shape is not None and len(shape) == 0:
            shape = (1,)
        return incoming_sample, shape

    def extend_if_has_space(
        self, incoming_samples: Union[List[Union[bytes, Sample, np.array]], np.array]
    ) -> int:
        self.prepare_for_write()
        if isinstance(incoming_samples, np.ndarray):
            return self._extend_if_has_space_numpy(incoming_samples)
        return self._extend_if_has_space_sequence(incoming_samples)

    def _extend_if_has_space_numpy(self, incoming_samples: np.array):
        num_samples = 0
        buffer_size = 0
        for incoming_sample in incoming_samples:
            shape = incoming_sample.shape
            if len(shape) == 0:
                shape = (1,)
            self.num_dims = self.num_dims or len(shape)
            sample_nbytes = incoming_sample.nbytes
            check_sample_shape(shape, self.num_dims)
            check_sample_size(sample_nbytes, self.min_chunk_size, self.compression)
            if not self.can_fit_sample(sample_nbytes, buffer_size):
                break
            buffer_size += sample_nbytes
            self.register_sample_to_headers(sample_nbytes, shape)
            self.tensor_meta.length += 1
            self.tensor_meta.update_shape_interval(shape)
            num_samples += 1

        buffer = incoming_samples[:num_samples].tobytes()
        self.data_bytes += buffer
        return num_samples

    def _extend_if_has_space_sequence(
        self, incoming_samples: List[Union[bytes, Sample, np.array]]
    ):
        num_samples = 0
        for incoming_sample in incoming_samples:
            serialized_sample, shape = self.serialize_sample(incoming_sample)
            self.num_dims = self.num_dims or len(shape)
            sample_nbytes = len(serialized_sample)
            check_sample_shape(shape, self.num_dims)
            check_sample_size(sample_nbytes, self.min_chunk_size, self.compression)
            if not self.can_fit_sample(sample_nbytes):
                break
            self.data_bytes += serialized_sample
            self.register_sample_to_headers(sample_nbytes, shape)
            self.tensor_meta.length += 1
            self.tensor_meta.update_shape_interval(shape)
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

        if copy:
            buffer = bytes(buffer)
        # print(len(buffer), self.dtype)
        return np.frombuffer(buffer, dtype=self.dtype).reshape(shape)

    def update_sample(
        self, local_sample_index: int, new_buffer: memoryview, new_shape: Tuple[int]
    ):
        raise NotImplementedError
