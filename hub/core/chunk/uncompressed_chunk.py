from typing import List, Optional, Tuple, Union

from numpy.core.numerictypes import nbytes
from hub.core.sample import Sample
from hub.core.serialize import (
    check_input_samples,
    bytes_to_text,
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
        shapes = []
        nbytes = []
        buffer = bytearray()
        num_samples = 0
        if isinstance(incoming_samples, np.ndarray):
            for i, incoming_sample in enumerate(incoming_samples):
                shape = incoming_sample.shape
                sample_nbytes = incoming_sample.nbytes
                if not self.can_fit_sample(sample_nbytes, sum(nbytes)):
                    break
                num_samples += 1
                shapes.append(shape)
                nbytes.append(sample_nbytes)
            buffer = incoming_samples[:num_samples].tobytes()
        else:
            for incoming_sample in incoming_samples:
                serialized_sample, shape = self.serialize_sample(incoming_sample)
                sample_nbytes = len(serialized_sample)
                if not self.can_fit_sample(sample_nbytes, sum(nbytes)):
                    break
                buffer += serialized_sample
                num_samples += 1
                shapes.append(shape)
                nbytes.append(sample_nbytes)
            
        check_input_samples(nbytes, shapes, self.min_chunk_size, None)
        self.shapes.extend(shapes)
        self.data_bytes += buffer
        for i in range(num_samples):
            self.register_sample_to_headers(nbytes[i], shapes[i])
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
        return np.frombuffer(buffer, dtype=self.dtype).reshape(shape)

    def update_sample(
        self, local_sample_index: int, new_buffer: memoryview, new_shape: Tuple[int]
    ):
        raise NotImplementedError
