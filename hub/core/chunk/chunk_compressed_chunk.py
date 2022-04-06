import numpy as np
from typing import List
from hub.core.compression import (
    compress_bytes,
    compress_multiple,
    decompress_bytes,
    decompress_multiple,
)
from hub.core.fast_forwarding import ffw_chunk
from hub.core.serialize import bytes_to_text, check_sample_shape
from hub.core.partial_sample import PartialSample
from hub.core.tiling.sample_tiles import SampleTiles
from hub.util.casting import intelligent_cast
from hub.util.compression import get_compression_ratio
from .base_chunk import BaseChunk, InputSample
from hub.core.serialize import infer_chunk_num_bytes
import hub


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
        self._compression_ratio = 0.5

    def extend_if_has_space(self, incoming_samples: List[InputSample]) -> float:  # type: ignore
        self.prepare_for_write()
        if self.is_byte_compression:
            return self.extend_if_has_space_byte_compression(incoming_samples)
        return self.extend_if_has_space_image_compression(incoming_samples)

    def extend_if_has_space_byte_compression(self, incoming_samples: List[InputSample]):
        num_samples = 0
        for i, incoming_sample in enumerate(incoming_samples):
            serialized_sample, shape = self.serialize_sample(
                incoming_sample,
                chunk_compression=self.compression,
                store_uncompressed_tiles=True,
            )

            self.num_dims = self.num_dims or len(shape)
            check_sample_shape(shape, self.num_dims)

            if isinstance(serialized_sample, SampleTiles):
                incoming_samples[i] = serialized_sample  # type: ignore
                if self.is_empty:
                    self.write_tile(serialized_sample)
                    num_samples += 0.5  # type: ignore
                    tile = serialized_sample.yield_uncompressed_tile()
                    if tile is not None:
                        self.decompressed_bytes = tile.tobytes()
                    self._changed = True
                break
            sample_nbytes = len(serialized_sample)

            recompressed = False  # This flag helps avoid double concatenation
            if (
                len(self.decompressed_bytes) + sample_nbytes  # type: ignore
            ) * self._compression_ratio > self.min_chunk_size:
                new_decompressed = self.decompressed_bytes + serialized_sample  # type: ignore
                compressed_bytes = compress_bytes(
                    new_decompressed, compression=self.compression
                )
                if len(compressed_bytes) > self.min_chunk_size:
                    break
                recompressed = True
                self.decompressed_bytes = new_decompressed
                self._compression_ratio /= 2
                self._data_bytes = compressed_bytes
                self._changed = False
            if not recompressed:
                self.decompressed_bytes += serialized_sample  # type: ignore
            self._changed = True
            self.register_in_meta_and_headers(sample_nbytes, shape)
            num_samples += 1
        return num_samples

    def extend_if_has_space_image_compression(
        self, incoming_samples: List[InputSample]
    ):
        num_samples = 0
        num_decompressed_bytes = sum(
            x.nbytes for x in self.decompressed_samples  # type: ignore
        )
        for i, incoming_sample in enumerate(incoming_samples):
            incoming_sample, shape = self.process_sample_img_compr(incoming_sample)

            if isinstance(incoming_sample, SampleTiles):
                incoming_samples[i] = incoming_sample  # type: ignore
                if self.is_empty:
                    self.write_tile(incoming_sample)
                    num_samples += 0.5  # type: ignore
                    tile = incoming_sample.yield_uncompressed_tile()
                    if tile is not None:
                        self.decompressed_samples = [tile]
                    self._changed = True
                break
            if (
                num_decompressed_bytes + incoming_sample.nbytes  # type: ignore
            ) * self._compression_ratio > self.min_chunk_size:
                compressed_bytes = compress_multiple(
                    self.decompressed_samples + [incoming_sample],  # type: ignore
                    compression=self.compression,
                )
                if len(compressed_bytes) > self.min_chunk_size:
                    break
                self._compression_ratio /= 2
                self._data_bytes = compressed_bytes
                self._changed = False

            shape = incoming_sample.shape  # type: ignore
            shape = self.normalize_shape(shape)

            self.num_dims = self.num_dims or len(shape)
            check_sample_shape(shape, self.num_dims)
            self.decompressed_samples.append(incoming_sample)  # type: ignore
            self._changed = True
            # Byte positions are not relevant for chunk wise image compressions, so incoming_num_bytes=None.
            self.register_in_meta_and_headers(None, shape)
            num_samples += 1
        return num_samples

    def _get_partial_sample_tile(self, as_bytes=None):
        if self.decompressed_samples or self.decompressed_bytes:
            return None
        if as_bytes is None:
            as_bytes = self.is_byte_compression
        return super(ChunkCompressedChunk, self)._get_partial_sample_tile(
            as_bytes=as_bytes
        )

    def read_sample(
        self,
        local_index: int,
        cast: bool = True,
        copy: bool = False,
        decompress: bool = True,
    ):
        if not decompress:
            raise NotImplementedError(
                "`decompress=False` is not supported by chunk compressed chunks as it can cause recompression."
            )
        partial_sample_tile = self._get_partial_sample_tile(as_bytes=False)
        if partial_sample_tile is not None:
            return partial_sample_tile
        if self.is_image_compression:
            return self.decompressed_samples[local_index]  # type: ignore

        shape = self.shapes_encoder[local_index]
        decompressed = memoryview(self.decompressed_bytes)  # type: ignore
        if not self.byte_positions_encoder.is_empty():
            sb, eb = self.byte_positions_encoder[local_index]
            decompressed = decompressed[sb:eb]
        if self.is_text_like:
            return bytes_to_text(decompressed, self.htype)
        return np.frombuffer(decompressed, dtype=self.dtype).reshape(shape)

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
        partial_sample_tile = self._get_partial_sample_tile()
        if partial_sample_tile is not None:
            self.decompressed_bytes = partial_sample_tile
        decompressed_buffer = self.decompressed_bytes

        new_data_uncompressed = self.create_updated_data(
            local_index, decompressed_buffer, serialized_sample
        )
        self.decompressed_bytes = new_data_uncompressed
        self._changed = True
        new_nb = (
            None if self.byte_positions_encoder.is_empty() else len(serialized_sample)
        )
        self.update_in_meta_and_headers(local_index, new_nb, shape)

    def update_sample_img_compression(self, local_index: int, new_sample: InputSample):
        new_sample = intelligent_cast(new_sample, self.dtype, self.htype)
        shape = new_sample.shape
        shape = self.normalize_shape(shape)
        self.check_shape_for_update(local_index, shape)
        partial_sample_tile = self._get_partial_sample_tile()
        if partial_sample_tile is not None:
            self.decompressed_samples = [partial_sample_tile]
        decompressed_samples = self.decompressed_samples

        decompressed_samples[local_index] = new_sample  # type: ignore
        self._changed = True
        self.update_in_meta_and_headers(local_index, None, shape)  # type: ignore

        self.data_bytes = bytearray(  # type: ignore
            compress_multiple(decompressed_samples, self.compression)  # type: ignore
        )
        self.update_in_meta_and_headers(local_index, None, shape)

    def process_sample_img_compr(self, sample):
        if isinstance(sample, SampleTiles):
            return sample, sample.tile_shape
        elif isinstance(sample, PartialSample):
            return (
                SampleTiles(
                    compression=self.compression,
                    chunk_size=self.min_chunk_size,
                    htype=self.htype,
                    dtype=self.dtype,
                    sample_shape=sample.sample_shape,
                    tile_shape=sample.tile_shape,
                ),
                sample.sample_shape,
            )
        if isinstance(sample, hub.core.tensor.Tensor):
            sample = sample.numpy()
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

    def _pop_sample(self):
        if self.is_byte_compression:
            self.decompressed_bytes = self.decompressed_bytes[
                : self.byte_positions_encoder[-1][0]
            ]
            self._data_bytes = compress_bytes(self.decompressed_bytes, self.compression)
        else:
            self.decompressed_samples.pop()
            self._data_bytes = compress_multiple(
                self.decompressed_samples, self.compression
            )
        self._changed = False

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

    @data_bytes.setter
    def data_bytes(self, value):
        self._data_bytes = value

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
            len_data=min(self.num_uncompressed_bytes, self.max_chunk_size),
        )

    def prepare_for_write(self):
        ffw_chunk(self)
        self.is_dirty = True
