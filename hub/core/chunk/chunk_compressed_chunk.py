import numpy as np
from typing import List, Sequence, Union
from hub.core.compression import (
    compress_bytes,
    compress_multiple,
    decompress_bytes,
    decompress_multiple,
)
from hub.core.serialize import (
    bytes_to_text,
    check_sample_shape,
)
from hub.core.tiling.sample_tiles import SampleTiles  # type: ignore
from hub.util.casting import intelligent_cast
from hub.util.exceptions import SampleDecompressionError
from .base_chunk import BaseChunk, InputSample


class ChunkCompressedChunk(BaseChunk):
    def extend_if_has_space(
        self, incoming_samples: Union[Sequence[InputSample], np.ndarray]
    ) -> float:
        self.prepare_for_write()
        if self.is_byte_compression:
            return self.extend_if_has_space_byte_compression(incoming_samples)
        return self.extend_if_has_space_image_compression(incoming_samples)

    def extend_if_has_space_byte_compression(
        self, incoming_samples: Union[Sequence[InputSample], np.ndarray]
    ) -> float:
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
                if isinstance(incoming_samples, List):
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
                if len(compressed_bytes) > self.min_chunk_size:
                    break

                self.data_bytes = compressed_bytes
                self.register_in_meta_and_headers(sample_nbytes, shape)
                num_samples += 1
                self._decompressed_bytes = buffer
        return num_samples

    def extend_if_has_space_image_compression(
        self, incoming_samples: Union[Sequence[InputSample], np.ndarray]
    ) -> float:
        num_samples: float = 0
        buffer_list = self.decompressed_samples if self.data_bytes else []

        for i, incoming_sample in enumerate(incoming_samples):
            if isinstance(incoming_sample, SampleTiles):
                if self.is_empty:
                    self.write_tile(incoming_sample, skip_bytes=True)
                    num_samples += 0.5
                    tile = incoming_sample.yield_uncompressed_tile()
                    self._decompressed_samples = tile
                break

            incoming_sample = intelligent_cast(incoming_sample, self.dtype, self.htype)
            shape = incoming_sample.shape
            shape = self.normalize_shape(shape)
            self.num_dims = self.num_dims or len(shape)
            check_sample_shape(shape, self.num_dims)
            buffer_list.append(incoming_sample)

            compressed_bytes = compress_multiple(buffer_list, self.compression)

            if len(compressed_bytes) > self.min_chunk_size:
                if self.is_empty and isinstance(incoming_samples, List):
                    incoming_samples[i] = SampleTiles(
                        incoming_sample,
                        self.compression,
                        self.min_chunk_size,
                        store_uncompressed_tiles=True,
                    )
                break

            self.data_bytes = compressed_bytes
            # Byte positions are not relevant for chunk wise image compressions, so incoming_num_bytes=None.
            self.register_in_meta_and_headers(None, shape)
            num_samples += 1
            self._decompressed_samples = buffer_list

        return num_samples

    @property
    def decompressed_samples(self) -> List[np.ndarray]:
        """Applicable only for compressed chunks. Returns samples contained in this chunk as a list of numpy arrays."""
        if not self._decompressed_samples:
            shapes = [
                self.shapes_encoder[i] for i in range(self.shapes_encoder.num_samples)
            ]
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

    def read_sample(
        self, local_sample_index: int, cast: bool = True, copy: bool = False
    ):
        if self.is_image_compression:
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
        serialized_sample, shape = self.serialize_sample(
            new_sample, chunk_compression=self.compression, break_into_tiles=False
        )
        self.check_shape_for_update(local_sample_index, shape)

        new_nb = len(serialized_sample)
        decompressed_buffer = self.decompressed_bytes

        new_data_uncompressed = self.create_buffer_with_updated_data(
            local_sample_index, decompressed_buffer, serialized_sample
        )
        self.data_bytes = bytearray(
            compress_bytes(new_data_uncompressed, compression=self.compression)
        )
        self._decompressed_bytes = new_data_uncompressed
        self.update_in_meta_and_headers(local_sample_index, new_nb, shape)

    def update_sample_image_compression(
        self,
        local_sample_index: int,
        new_sample: InputSample,
    ):
        new_sample = intelligent_cast(new_sample, self.dtype, self.htype)
        shape = new_sample.shape
        shape = self.normalize_shape(shape)
        self.check_shape_for_update(local_sample_index, shape)
        decompressed_samples = self.decompressed_samples
        decompressed_samples[local_sample_index] = new_sample
        self.data_bytes = bytearray(
            compress_multiple(decompressed_samples, self.compression)
        )
        self.update_in_meta_and_headers(local_sample_index, None, shape)
