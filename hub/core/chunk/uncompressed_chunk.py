from typing import List, Optional, Tuple, Union
from hub.core.sample import Sample
from hub.core.serialize import (
    _check_input_samples_are_valid,
    bytes_to_text,
    serialize_numpy_and_base_types,
    text_to_bytes,
)
from .base_chunk import BaseChunk
import numpy as np

SampleValue = Union[bytes, Sample, np.ndarray, int, float, bool, dict, list, str]
SerializedOutput = tuple(bytes, Optional[tuple])


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
        if isinstance(incoming_samples, np.array):
            # optimized to directly write bytes of multiple arrays at once
            for i, incoming_sample in enumerate(incoming_samples):
                if incoming_sample.nbytes + self.num_data_bytes > self.max_chunk_size:
                    self.data_bytes += incoming_samples[:i].tobytes()
                    self.shapes.extend([incoming_sample.shape] * (i - 1))
                    for _ in range(i):
                        self.register_sample_to_headers(
                            incoming_sample.nbytes, incoming_sample.shape
                        )
                    return i
        else:
            for i, incoming_sample in enumerate(incoming_samples):
                serialized_sample, shape = self.serialize_sample(incoming_sample)
                sample_nbytes = len(serialized_sample)
                if self.num_data_bytes + sample_nbytes > self.max_chunk_size:
                    return i
                self.data_bytes += serialized_sample
                self.shapes.append(shape)
                self.register_sample_to_headers(sample_nbytes, shape)
        return len(incoming_samples)

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
        return np.frombuffer(buffer, dtype=self.dtype).reshape(shape)

    def update_sample(
        self, local_sample_index: int, new_buffer: memoryview, new_shape: Tuple[int]
    ):
        raise NotImplementedError
