import os
import struct
from typing import List, Optional, Union
from deeplake.core.compression import decompress_array, decompress_bytes
from deeplake.core.sample import Sample  # type: ignore
from deeplake.core.serialize import (
    check_sample_shape,
    bytes_to_text,
)
from deeplake.core.tiling.sample_tiles import SampleTiles
from deeplake.core.polygon import Polygons
from deeplake.util.video import normalize_index
from .base_chunk import BaseChunk, InputSample, catch_chunk_read_error


class SampleCompressedChunk(BaseChunk):
    def extend_if_has_space(self, incoming_samples: List[InputSample], update_tensor_meta: bool = True, ignore_errors: bool = False, **kwargs) -> float:  # type: ignore
        self.prepare_for_write()
        num_samples: float = 0
        dtype = self.dtype if self.is_byte_compression else None
        compr = self.compression
        skipped: List[int] = []

        for i, incoming_sample in enumerate(incoming_samples):
            try:
                serialized_sample, shape = self.serialize_sample(incoming_sample, compr)
                if shape is not None:
                    self.num_dims = self.num_dims or len(shape)
                    check_sample_shape(shape, self.num_dims)
            except Exception:
                if ignore_errors:
                    skipped.append(i)
                    continue
                raise

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

                    self.register_in_meta_and_headers(
                        sample_nbytes,
                        shape,
                        update_tensor_meta=update_tensor_meta,
                    )
                    num_samples += 1
                else:
                    if serialized_sample:
                        sample = Sample(
                            buffer=serialized_sample, compression=compr, shape=shape, dtype=dtype  # type: ignore
                        )
                        sample.htype = self.htype
                        incoming_samples[i] = sample
                    break

        for i in reversed(skipped):
            incoming_samples.pop(i)
        return num_samples

    @catch_chunk_read_error
    def read_sample(  # type: ignore
        self,
        local_index: int,
        cast: bool = True,
        copy: bool = False,
        sub_index: Optional[Union[int, slice]] = None,
        stream: bool = False,
        decompress: bool = True,
        is_tile: bool = False,
        to_pil: bool = False,
    ):
        self.check_empty_before_read()
        partial_sample_tile = self._get_partial_sample_tile()
        if partial_sample_tile is not None:
            return partial_sample_tile
        buffer = self.memoryview_data
        bps = self.byte_positions_encoder
        bps_empty = bps.is_empty()
        if not bps_empty:
            sb, eb = bps[local_index]
            if stream and self.is_video_compression:
                header_size = struct.unpack("<i", buffer[-4:])[
                    0
                ]  # last 4 bytes store size of header

                # create subfile url to pass to ffmpeg. header_size + sb will give starting point of video bytes.
                # https://ffmpeg.org/ffmpeg-protocols.html#subfile
                buffer = (
                    f"subfile,,start,{header_size + sb},end,{header_size + eb},,:"
                    + bytes(buffer[:-4]).decode("utf-8")
                )
                if not decompress:
                    return buffer
            else:
                buffer = buffer[sb:eb]
        if not decompress:
            return bytes(buffer) if copy else buffer
        if not is_tile and self.is_fixed_shape:
            shape = tuple(self.tensor_meta.min_shape)
        else:
            try:
                shape = self.shapes_encoder[local_index]
            except IndexError as e:
                if not bps_empty:
                    self.num_dims = self.num_dims or len(self.tensor_meta.max_shape)
                    shape = (0,) * self.num_dims
                else:
                    raise e

        nframes = shape[0]
        if self.is_text_like:
            buffer = decompress_bytes(buffer, compression=self.compression)
            buffer = bytes(buffer)
            return bytes_to_text(buffer, self.htype)

        squeeze = isinstance(sub_index, int)

        start, stop, step, reverse = normalize_index(sub_index, nframes)

        if start > nframes:
            raise IndexError("Start index out of bounds.")

        if self.tensor_meta.htype == "polygon":
            buffer = decompress_bytes(buffer, self.compression)
            return Polygons.frombuffer(
                bytes(buffer),
                dtype=self.tensor_meta.dtype,
                ndim=shape[-1],
            )

        sample = decompress_array(
            buffer,
            shape,
            self.dtype,
            self.compression,
            start_idx=start,
            end_idx=stop,
            step=step,
            reverse=reverse,
            to_pil=to_pil,
        )
        if to_pil:
            return sample

        if squeeze:
            sample = sample.squeeze(0)

        if cast and sample.dtype != self.dtype:
            sample = sample.astype(self.dtype)
        elif copy and not sample.flags["WRITEABLE"]:
            sample = sample.copy()
        return sample

    def update_sample(self, local_index: int, sample: InputSample):
        self.prepare_for_write()
        serialized_sample, shape = self.serialize_sample(
            sample, self.compression, break_into_tiles=False
        )

        self.check_shape_for_update(shape)
        old_data = self.data_bytes
        self.data_bytes = self.create_updated_data(
            local_index, old_data, serialized_sample
        )

        # update encoders and meta
        new_nb = (
            None if self.byte_positions_encoder.is_empty() else len(serialized_sample)
        )
        self.update_in_meta_and_headers(local_index, new_nb, shape)
