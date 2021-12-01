import numpy as np
from typing import List, Optional, Sequence, Union
from hub.core.compression import (
    compress_bytes,
    compress_multiple,
    decompress_bytes,
    decompress_multiple,
)
from hub.core.sample import Sample  # type: ignore
from hub.core.fast_forwarding import ffw_chunk
from hub.core.serialize import (
    bytes_to_text,
    check_sample_shape,
)
from hub.util.casting import intelligent_cast
from hub.util.exceptions import SampleDecompressionError
from .base_chunk import BaseChunk, InputSample
from hub.core.serialize import infer_chunk_num_bytes


class ChunkCompressedChunk(BaseChunk):
    def __init__(self, *args, **kwargs):
        super(ChunkCompressedChunk, self).__init__(*args, **kwargs)
        if self.is_byte_compression:
            self.decompressed_bytes = bytearray(
                decompress_bytes(self._data_bytes, self.compression)
            )
        else:
            shapes = [
                self.shapes_encoder[i] for i in range(self.shapes_encoder.num_samples)
            ]
            self.decompressed_samples = decompress_multiple(self._data_bytes, shapes)
        self._changed = False
        self._compression_ratio = 1

    def extend_if_has_space(
        self, incoming_samples: Union[Sequence[InputSample], np.ndarray]
    ) -> int:
        self.prepare_for_write()
        if self.is_byte_compression:
            return self.extend_if_has_space_byte_compression(incoming_samples)
        return self.extend_if_has_space_image_compression(incoming_samples)

    def extend_if_has_space_byte_compression(
        self, incoming_samples: Union[Sequence[InputSample], np.ndarray]
    ):
        num_samples = 0
        for incoming_sample in incoming_samples:
            serialized_sample, shape = self.sample_to_bytes(
                incoming_sample, None, False
            )
            sample_nbytes = len(serialized_sample)

            recompressed = False  # This flag helps avoid double concatenation
            if (
                len(self.decompressed_bytes) + sample_nbytes
            ) / self._compression_ratio > self.min_chunk_size:
                new_decompressed = self.decompressed_bytes + serialized_sample
                compressed_bytes = compress_bytes(
                    new_decompressed, compression=self.compression
                )
                if len(compressed_bytes) > self.min_chunk_size:
                    break
                recompressed = True
                self.decompressed_bytes = new_decompressed
                self._compression_ratio *= 2
                self._data_bytes = compressed_bytes
                self._changed = False

            self.num_dims = self.num_dims or len(shape)
            check_sample_shape(shape, self.num_dims)
            if not recompressed:
                self.decompressed_bytes += serialized_sample
            self._changed = True
            self.register_in_meta_and_headers(sample_nbytes, shape)
            num_samples += 1
        return num_samples

    def extend_if_has_space_image_compression(
        self, incoming_samples: Union[Sequence[InputSample], np.ndarray]
    ):
        num_samples = 0
        num_decompressed_bytes = sum(
            x.nbytes for x in self.decompressed_samples
        )  # TODO cache this
        for incoming_sample in incoming_samples:
            if isinstance(incoming_sample, bytes):
                raise ValueError(
                    "Chunkwise image compression is not applicable on bytes."
                )
            incoming_sample = intelligent_cast(incoming_sample, self.dtype, self.htype)
            if isinstance(incoming_sample, Sample):
                incoming_sample = incoming_sample.array
            if (
                num_decompressed_bytes + incoming_sample.nbytes
            ) / self._compression_ratio > self.min_chunk_size:
                compressed_bytes = compress_multiple(
                    self.decompressed_samples + [incoming_sample],
                    compression=self.compression,
                )
                if len(compressed_bytes) > self.min_chunk_size:
                    break
                self._compression_ratio *= 2
                self._data_bytes = compressed_bytes
                self._changed = False

            shape = incoming_sample.shape
            shape = self.normalize_shape(shape)

            self.num_dims = self.num_dims or len(shape)
            check_sample_shape(shape, self.num_dims)
            self.decompressed_samples.append(incoming_sample)
            self._changed = True
            # Byte positions are not relevant for chunk wise image compressions, so incoming_num_bytes=None.
            self.register_in_meta_and_headers(None, shape)
            num_samples += 1
        return num_samples

    def read_sample(
        self, local_sample_index: int, cast: bool = True, copy: bool = False
    ):
        if not self.is_byte_compression:
            return self.decompressed_samples[local_sample_index]

        sb, eb = self.byte_positions_encoder[local_sample_index]
        shape = self.shapes_encoder[local_sample_index]
        decompressed = memoryview(self.decompressed_bytes)
        buffer = decompressed[sb:eb]
        if self.is_text_like:
            return bytes_to_text(buffer, self.htype)
        return np.frombuffer(decompressed[sb:eb], dtype=self.dtype).reshape(shape)

    def update_sample(
        self,
        local_sample_index: int,
        new_sample: InputSample,
    ):
        self.prepare_for_write()
        if self.is_byte_compression:
            self.update_sample_byte_compression(local_sample_index, new_sample)
        else:
            self.update_sample_image_compression(local_sample_index, new_sample)

    def update_sample_byte_compression(
        self,
        local_sample_index: int,
        new_sample: InputSample,
    ):
        serialized_sample, shape = self.sample_to_bytes(new_sample, None, False)
        self.check_shape_for_update(local_sample_index, shape)

        new_nb = len(serialized_sample)
        decompressed_buffer = self.decompressed_bytes

        new_data_uncompressed = self.create_buffer_with_updated_data(
            local_sample_index, decompressed_buffer, serialized_sample
        )
        self.decompressed_bytes = new_data_uncompressed
        self._changed = True
        self.update_in_meta_and_headers(local_sample_index, new_nb, shape)

    def update_sample_image_compression(
        self,
        local_sample_index: int,
        new_sample: InputSample,
    ):
        new_sample = intelligent_cast(new_sample, self.dtype, self.htype)
        if isinstance(new_sample, Sample):
            new_sample = new_sample.array
        shape = new_sample.shape
        shape = self.normalize_shape(shape)
        self.check_shape_for_update(local_sample_index, shape)
        decompressed_samples = self.decompressed_samples
        decompressed_samples[local_sample_index] = new_sample
        self._changed = True
        self.update_in_meta_and_headers(local_sample_index, None, shape)

    def _compress(self):
        if self.is_byte_compression:
            self._data_bytes = compress_bytes(self.decompressed_bytes, self.compression)
        else:
            self._data_bytes = compress_multiple(
                self.decompressed_samples, self.compression
            )

    @property
    def data_bytes(self):
        if self._changed:
            self._compress()
            self._changed = False
        return self._data_bytes

    @property
    def num_uncompressed_bytes(self):
        if self.is_byte_compression:
            return len(self.decompressed_bytes)
        return sum(x.nbytes for x in self.decompressed_samples)

    @property
    def nbytes(self):
        """Calculates the number of bytes `tobytes` will be without having to call `tobytes`. Used by `LRUCache` to determine if this chunk can be cached."""
        return infer_chunk_num_bytes(
            self.version,
            self.shapes_encoder.array,
            self.byte_positions_encoder.array,
            len_data=min(self.num_uncompressed_bytes // 4, 100),
        )

    def prepare_for_write(self):
        ffw_chunk(self)
