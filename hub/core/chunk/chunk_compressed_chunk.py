import numpy as np
from typing import List
from hub.core.compression import (
    compress_bytes,
    compress_multiple,
    decompress_bytes,
    decompress_multiple,
)
from hub.core.serialize import bytes_to_text, check_sample_shape
from hub.core.tiling.sample_tiles import SampleTiles
from hub.util.casting import intelligent_cast
from hub.util.compression import get_compression_ratio
from hub.util.exceptions import SampleDecompressionError
from .base_chunk import BaseChunk, InputSample


class ChunkCompressedChunk(BaseChunk):
    def extend_if_has_space(self, incoming_samples: List[InputSample]) -> float:
        self.prepare_for_write()
        if self.is_byte_compression:
            return self.extend_if_has_space_byte_compression(incoming_samples)
        return self.extend_if_has_space_img_compression(incoming_samples)

    def extend_if_has_space_byte_compression(self, incoming_samples: List[InputSample]):
        num_samples: float = 0
        buffer = bytearray(self.decompressed_bytes) if self.data_bytes else bytearray()
        for i, incoming_sample in enumerate(incoming_samples):
            serialized_sample, shape = self.serialize_sample(
                incoming_sample,
                chunk_compression=self.compression,
                store_uncompressed_tiles=True,
            )
            self.num_dims = self.num_dims or len(shape)
            check_sample_shape(shape, self.num_dims)

            if isinstance(serialized_sample, SampleTiles):
                incoming_samples[i] = serialized_sample
                if self.is_empty:
                    self.write_tile(serialized_sample)
                    num_samples += 0.5
                    tile = serialized_sample.yield_uncompressed_tile()
                    self._decompressed_bytes = tile.tobytes()
                break
            else:
                sample_nbytes = len(serialized_sample)
                buffer += serialized_sample
                compressed_bytes = compress_bytes(buffer, self.compression)
                if self.is_empty or len(compressed_bytes) < self.min_chunk_size:
                    self.data_bytes = compressed_bytes
                    self.register_in_meta_and_headers(sample_nbytes, shape)
                    num_samples += 1
                    self._decompressed_bytes = buffer
                else:
                    break
        return num_samples

    def extend_if_has_space_img_compression(self, incoming_samples: List[InputSample]):
        num_samples: float = 0
        buffer_list = self.decompressed_samples if self.data_bytes else []

        for i, incoming_sample in enumerate(incoming_samples):
            incoming_sample, shape = self.process_sample_img_compr(incoming_sample)

            if isinstance(incoming_sample, SampleTiles):
                incoming_samples[i] = incoming_sample
                if self.is_empty:
                    self.write_tile(incoming_sample, skip_bytes=True)
                    num_samples += 0.5
                    tile = incoming_sample.yield_uncompressed_tile()
                    self._decompressed_samples = [tile]
                break

            buffer_list.append(incoming_sample)
            compressed_bytes = compress_multiple(buffer_list, self.compression)
            if self.is_empty or len(compressed_bytes) < self.min_chunk_size:
                self.data_bytes = compressed_bytes
                # Byte positions aren't relevant for chunk wise img compressions
                self.register_in_meta_and_headers(None, shape)
                num_samples += 1
                self._decompressed_samples = buffer_list
            else:
                buffer_list.pop()
                break

        return num_samples

    @property
    def decompressed_samples(self) -> List[np.ndarray]:
        """Applicable only for compressed chunks. Returns samples contained in this chunk as a list of numpy arrays."""
        if self._decompressed_samples is None:
            num_samples = self.shapes_encoder.num_samples
            shapes = [self.shapes_encoder[i] for i in range(num_samples)]
            self._decompressed_samples = decompress_multiple(self.data_bytes, shapes)
        return self._decompressed_samples

    @property
    def decompressed_bytes(self) -> bytes:
        """Applicable only for chunks compressed using a byte compression. Returns the contents of the chunk as a decompressed buffer."""
        if self._decompressed_bytes is None:
            try:
                data = decompress_bytes(self.data_bytes, self.compression)
                self._decompressed_bytes = data
            except SampleDecompressionError:
                raise ValueError(
                    "Chunk.decompressed_bytes can not be called on chunks compressed with image compressions. Use Chunk.get_samples() instead."
                )
        return self._decompressed_bytes

    def read_sample(self, local_index: int, cast: bool = True, copy: bool = False):
        if self.is_image_compression:
            return self.decompressed_samples[local_index]

        sb, eb = self.byte_positions_encoder[local_index]
        shape = self.shapes_encoder[local_index]
        decompressed = memoryview(self.decompressed_bytes)
        buffer = decompressed[sb:eb]
        if self.is_text_like:
            return bytes_to_text(buffer, self.htype)
        return np.frombuffer(decompressed[sb:eb], dtype=self.dtype).reshape(shape)

    def update_sample(self, local_index: int, new_sample: InputSample):
        self.prepare_for_write()
        if self.is_byte_compression:
            self.update_sample_byte_compression(local_index, new_sample)
        else:
            self.update_sample_img_compression(local_index, new_sample)

    def update_sample_byte_compression(self, local_index: int, new_sample: InputSample):
        serialized_sample, shape = self.serialize_sample(
            new_sample, chunk_compression=self.compression, break_into_tiles=False
        )
        self.check_shape_for_update(local_index, shape)

        new_nb = len(serialized_sample)
        decompressed_buffer = self.decompressed_bytes

        new_data_uncompressed = self.create_updated_data(
            local_index, decompressed_buffer, serialized_sample
        )
        self.data_bytes = bytearray(
            compress_bytes(new_data_uncompressed, self.compression)
        )
        self._decompressed_bytes = new_data_uncompressed
        self.update_in_meta_and_headers(local_index, new_nb, shape)

    def update_sample_img_compression(self, local_index: int, new_sample: InputSample):
        new_sample = intelligent_cast(new_sample, self.dtype, self.htype)
        shape = new_sample.shape
        shape = self.normalize_shape(shape)
        self.check_shape_for_update(local_index, shape)
        decompressed_samples = self.decompressed_samples
        decompressed_samples[local_index] = new_sample
        self.data_bytes = bytearray(
            compress_multiple(decompressed_samples, self.compression)
        )
        self.update_in_meta_and_headers(local_index, None, shape)

    def process_sample_img_compr(self, sample):
        if isinstance(sample, SampleTiles):
            return sample, sample.tile_shape

        sample = intelligent_cast(sample, self.dtype, self.htype)
        shape = sample.shape
        shape = self.normalize_shape(shape)
        self.num_dims = self.num_dims or len(shape)
        check_sample_shape(shape, self.num_dims)

        ratio = get_compression_ratio(self.compression)
        approx_compressed_size = sample.nbytes * ratio

        if approx_compressed_size > self.min_chunk_size:
            sample = SampleTiles(
                sample,
                self.compression,
                self.min_chunk_size,
                store_uncompressed_tiles=True,
            )

        return sample, shape
