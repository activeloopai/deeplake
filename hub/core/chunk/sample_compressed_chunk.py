import os
import struct
from typing import List, Optional, Union
from hub.core.compression import decompress_array, decompress_bytes
from hub.core.sample import Sample  # type: ignore
from hub.core.serialize import (
    check_sample_shape,
    bytes_to_text,
)
from hub.core.tiling.sample_tiles import SampleTiles
from .base_chunk import BaseChunk, InputSample


class SampleCompressedChunk(BaseChunk):
    def extend_if_has_space(self, incoming_samples: List[InputSample]) -> float:  # type: ignore
        self.prepare_for_write()
        num_samples: float = 0
        dtype = self.dtype if self.is_byte_compression else None
        compr = self.compression

        for i, incoming_sample in enumerate(incoming_samples):
            serialized_sample, shape = self.serialize_sample(incoming_sample, compr)
            self.num_dims = self.num_dims or len(shape)
            check_sample_shape(shape, self.num_dims)

            if isinstance(serialized_sample, SampleTiles):
                incoming_samples[i] = serialized_sample
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
                    if serialized_sample:
                        buffer = serialized_sample
                        sample = Sample(
                            buffer=buffer, compression=compr, shape=shape, dtype=dtype
                        )
                        incoming_samples[i] = sample
                    break
        return num_samples

    def read_sample(
        self,
        local_index: int,
        cast: bool = True,
        copy: bool = False,
        sub_index: Optional[Union[int, slice]] = None,
        stream: bool = False,
    ):
        buffer = self.memoryview_data
        if not self.byte_positions_encoder.is_empty():
            sb, eb = self.byte_positions_encoder[local_index]
            if stream and self.is_video_compression:
                header_size = struct.unpack("<i", buffer[-4:])[
                    0
                ]  # last 4 bytes store size of header
                buffer = (
                    f"subfile,,start,{header_size + sb},end,{header_size + eb},,:"
                    + bytes(buffer[:-4]).decode("utf-8")
                )
            else:
                buffer = buffer[sb:eb]
        shape = self.shapes_encoder[local_index]
        if self.is_text_like:
            buffer = decompress_bytes(buffer, compression=self.compression)
            buffer = bytes(buffer)
            return bytes_to_text(buffer, self.htype)

        if sub_index is not None:
            if isinstance(sub_index, int):
                start = sub_index
                if sub_index >= 0:
                    end = start + 1
                else:
                    end = None
                squeeze = True
                step = 1
                reverse = False
            elif isinstance(sub_index, slice):
                start = sub_index.start
                end = sub_index.stop
                step = sub_index.step
                squeeze = False
                reverse = step is not None and step < 0
            sample = decompress_array(
                buffer,
                shape,
                self.dtype,
                self.compression,
                start_idx=start,
                end_idx=end,
                step=abs(step) if step is not None else 1,
                reverse=reverse,
            )
            if squeeze:
                sample = sample.squeeze(0)
        else:
            sample = decompress_array(buffer, shape, self.dtype, self.compression)
        if cast and sample.dtype != self.dtype:
            sample = sample.astype(self.dtype)
        return sample

    def update_sample(self, local_index: int, sample: InputSample):
        self.prepare_for_write()
        serialized_sample, shape = self.serialize_sample(
            sample, self.compression, break_into_tiles=False
        )

        self.check_shape_for_update(local_index, shape)
        old_data = self.data_bytes
        self.data_bytes = self.create_updated_data(
            local_index, old_data, serialized_sample
        )

        # update encoders and meta
        new_nb = (
            None if self.byte_positions_encoder.is_empty() else len(serialized_sample)
        )
        self.update_in_meta_and_headers(local_index, new_nb, shape)
