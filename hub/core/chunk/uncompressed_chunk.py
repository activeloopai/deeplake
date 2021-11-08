from typing import List, Optional, Tuple, Union
from numpy.core.numerictypes import nbytes
from hub.core.sample import Sample
from hub.core.serialize import (
    check_sample_shape,
    bytes_to_text,
    check_sample_size,
    text_to_bytes,
)
from hub.core.tiling.tile import SampleTiles
from hub.util.casting import intelligent_cast
from .base_chunk import BaseChunk
import numpy as np

SampleValue = Union[bytes, Sample, np.ndarray, int, float, bool, dict, list, str]
SerializedOutput = tuple[bytes, Optional[tuple]]

import numpy as np
from typing import List, Optional, Tuple, Union


class UncompressedChunk(BaseChunk):
    def needs_to_be_tiled(self, nbytes) -> bool:
        return nbytes > self.min_chunk_size

    def sample_to_bytes(
        self,
        incoming_sample: SampleValue,
        sample_compression: Optional[str],
        is_byte_compression,
    ) -> SerializedOutput:
        """Converts the sample into bytes"""
        dt, ht = self.dtype, self.htype
        if self.is_text_like:
            incoming_sample, shape = text_to_bytes(incoming_sample, dt, ht)
            # if sample_compression:
            #     incoming_sample = compress_bytes(incoming_sample, sample_compression)
        elif isinstance(incoming_sample, Sample):
            shape = incoming_sample.shape
            shape = self.convert_to_rgb(shape)
            # if sample_compression:
            #     if is_byte_compression:
            #         # Byte compressions don't store dtype, need to cast to expected dtype
            #         arr = intelligent_cast(incoming_sample.array, dt, ht)
            #         incoming_sample = Sample(array=arr)
            #     incoming_sample = incoming_sample.compressed_bytes(sample_compression)
            # else:
            incoming_sample = incoming_sample.array
            if incoming_sample.nbytes > self.min_chunk_size:
                incoming_sample = SampleTiles(incoming_sample, sample_compression, self.min_chunk_size)
            else:
                incoming_sample = incoming_sample.tobytes()
        elif isinstance(incoming_sample, bytes):
            shape = None
        elif isinstance(
            incoming_sample,
            (np.ndarray, list, int, float, bool, np.integer, np.floating, np.bool_),
        ):
            incoming_sample = intelligent_cast(incoming_sample, dt, ht)
            shape = incoming_sample.shape
            if incoming_sample.nbytes > self.min_chunk_size:
                incoming_sample = SampleTiles(incoming_sample, sample_compression, self.min_chunk_size)
            else:
                incoming_sample = incoming_sample.tobytes()
        elif isinstance(incoming_sample, SampleTiles):
            shape = incoming_sample.sample_shape
        else:
            raise TypeError(f"Cannot serialize sample of type {type(incoming_sample)}")
        shape = self.normalize_shape(shape)
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

    def _extend_if_has_space_sequence(
        self, incoming_samples: List[Union[bytes, Sample, np.array]]
    ):
        num_samples = 0
        for i, incoming_sample in enumerate(incoming_samples):
            serialized_sample, shape = self.sample_to_bytes(
                incoming_sample, None, False
            )
            self.num_dims = self.num_dims or len(shape)
            check_sample_shape(shape, self.num_dims)
            # check_sample_size(sample_nbytes, self.min_chunk_size, self.compression)
            if isinstance(serialized_sample, SampleTiles):
                incoming_samples[i] = serialized_sample
                if not self.data_bytes:
                    data = serialized_sample.yield_tile()
                    sample_nbytes = len(data)
                    self.data_bytes = data
                    tile_shape = serialized_sample.tile_shape
                    update_meta = serialized_sample.is_first_write()
                    self.register_sample_to_headers(sample_nbytes, tile_shape)
                    if update_meta:
                        self.tensor_meta.length += 1
                        self.tensor_meta.update_shape_interval(shape)
                    num_samples += 0.5
                break
            else:
                sample_nbytes = len(serialized_sample)
                if not self.can_fit_sample(sample_nbytes):
                    break
                self.data_bytes += serialized_sample
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

        if copy:
            buffer = bytes(buffer)
        return np.frombuffer(buffer, dtype=self.dtype).reshape(shape)

    def update_sample(
        self,
        local_sample_index: int,
        new_sample: Union[bytes, Sample, np.ndarray, int, float, bool, dict, list, str],
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
