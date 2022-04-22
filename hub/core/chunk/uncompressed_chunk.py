import numpy as np
from typing import List, Union
from hub.core.serialize import check_sample_shape, bytes_to_text
from hub.core.tiling.sample_tiles import SampleTiles
from hub.util.casting import intelligent_cast
from .base_chunk import BaseChunk, InputSample


class UncompressedChunk(BaseChunk):
    def extend_if_has_space(  # type: ignore
        self, incoming_samples: Union[List[InputSample], np.ndarray]
    ) -> float:
        self.prepare_for_write()
        if isinstance(incoming_samples, np.ndarray):
            return self._extend_if_has_space_numpy(incoming_samples)
        return self._extend_if_has_space_list(incoming_samples)

    def _extend_if_has_space_numpy(self, incoming_samples: np.ndarray) -> float:
        num_samples: int = 0
        buffer_size = 0

        for sample in incoming_samples:
            shape = self.normalize_shape(sample.shape)
            self.num_dims = self.num_dims or len(shape)
            sample_nbytes = sample.nbytes
            check_sample_shape(shape, self.num_dims)
            # no need to check sample size, this code is only reached when size smaller than chunk
            if not self.can_fit_sample(sample_nbytes, buffer_size):
                break
            buffer_size += sample_nbytes
            num_samples += 1

        samples = incoming_samples[:num_samples]
        samples = intelligent_cast(samples, self.dtype, self.htype)
        self.data_bytes += samples.tobytes()

        if num_samples > 0:
            shape = self.normalize_shape(samples[0].shape)
            sample_nbytes = samples[0].nbytes
            for _ in range(num_samples):
                self.register_in_meta_and_headers(sample_nbytes, shape)

        return float(num_samples)

    def _extend_if_has_space_list(self, incoming_samples: List[InputSample]) -> float:
        num_samples: float = 0

        for i, incoming_sample in enumerate(incoming_samples):
            serialized_sample, shape = self.serialize_sample(incoming_sample)
            self.num_dims = self.num_dims or len(shape)
            check_sample_shape(shape, self.num_dims)

            if isinstance(serialized_sample, SampleTiles):
                incoming_samples[i] = serialized_sample  # type: ignore
                if self.is_empty:
                    self.write_tile(serialized_sample)
                    num_samples += 0.5
                break
            else:
                sample_nbytes = len(serialized_sample)
                if self.is_empty or self.can_fit_sample(sample_nbytes):
                    self.data_bytes += serialized_sample  # type: ignore
                    self.register_in_meta_and_headers(sample_nbytes, shape)
                    num_samples += 1
                else:
                    break

        return num_samples

    def read_sample(
        self,
        local_index: int,
        cast: bool = True,
        copy: bool = False,
        decompress: bool = True,
    ):
        partial_sample_tile = self._get_partial_sample_tile()
        if partial_sample_tile is not None:
            return partial_sample_tile
        buffer = self.memoryview_data
        shape = self.shapes_encoder[local_index]
        if not self.byte_positions_encoder.is_empty():
            sb, eb = self.byte_positions_encoder[local_index]
            buffer = buffer[sb:eb]
        if not decompress:
            if copy:
                buffer = bytes(buffer)
            return buffer
        if self.is_text_like:
            buffer = bytes(buffer)
            return bytes_to_text(buffer, self.htype)
        buffer = bytes(buffer) if copy else buffer
        return np.frombuffer(buffer, dtype=self.dtype).reshape(shape)

    def update_sample(self, local_index: int, sample: InputSample):
        self.prepare_for_write()
        serialized_sample, shape = self.serialize_sample(sample, break_into_tiles=False)
        self.check_shape_for_update(local_index, shape)
        new_nb = (
            None if self.byte_positions_encoder.is_empty() else len(serialized_sample)
        )

        old_data = self.data_bytes
        self.data_bytes = self.create_updated_data(
            local_index, old_data, serialized_sample
        )

        self.update_in_meta_and_headers(local_index, new_nb, shape)
