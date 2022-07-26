import numpy as np
from typing import List
from hub.core.compression import (
    compress_bytes,
    compress_multiple,
    decompress_bytes,
    decompress_multiple,
)
from hub.core.fast_forwarding import ffw_chunk
from hub.core.meta.encode.shape import ShapeEncoder
from hub.core.serialize import bytes_to_text, check_sample_shape
from hub.core.partial_sample import PartialSample
from hub.core.tiling.sample_tiles import SampleTiles
from hub.util.casting import intelligent_cast
from hub.util.compression import get_compression_ratio
from hub.util.exceptions import EmptyTensorError
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

    def extend_if_has_space(self, incoming_samples: List[InputSample], update_tensor_meta: bool = True) -> float:  # type: ignore
        self.prepare_for_write()
        if self.is_byte_compression:
            return self.extend_if_has_space_byte_compression(
                incoming_samples, update_tensor_meta=update_tensor_meta
            )
        return self.extend_if_has_space_image_compression(
            incoming_samples, update_tensor_meta=update_tensor_meta
        )

    def extend_if_has_space_byte_compression(
        self,
        incoming_samples: List[InputSample],
        update_tensor_meta: bool = True,
    ):
        num_samples = 0

        for i, incoming_sample in enumerate(incoming_samples):
            serialized_sample, shape = self.serialize_sample(
                incoming_sample,
                chunk_compression=self.compression,
                store_uncompressed_tiles=True,
            )

            if shape is not None:
                self.num_dims = self.num_dims or len(shape)
                check_sample_shape(shape, self.num_dims)

            # for tiles we do not need to check
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

                decompressed_bytes = self.decompressed_bytes
                new_decompressed = self.decompressed_bytes + serialized_sample  # type: ignore

                compressed_bytes = compress_bytes(
                    new_decompressed, compression=self.compression
                )
                num_compressed_bytes = len(compressed_bytes)
                tiling_threshold = self.tiling_threshold
                if num_compressed_bytes > self.min_chunk_size and not (
                    not decompressed_bytes
                    and (
                        tiling_threshold < 0 or num_compressed_bytes > tiling_threshold
                    )
                ):
                    break
                recompressed = True
                self.decompressed_bytes = new_decompressed
                self._compression_ratio /= 2
                self._data_bytes = compressed_bytes
                self._changed = False
            if not recompressed:
                self.decompressed_bytes += serialized_sample  # type: ignore

            self._changed = True
            self.register_in_meta_and_headers(
                sample_nbytes, shape, update_tensor_meta=update_tensor_meta
            )
            num_samples += 1
        return num_samples

    def extend_if_has_space_image_compression(
        self,
        incoming_samples: List[InputSample],
        update_tensor_meta: bool = True,
    ):
        num_samples = 0
        num_decompressed_bytes = sum(
            x.nbytes for x in self.decompressed_samples  # type: ignore
        )

        for i, incoming_sample in enumerate(incoming_samples):
            incoming_sample, shape = self.process_sample_img_compr(incoming_sample)

            if shape is not None and self.is_empty_tensor and len(shape) != 3:
                self.change_dimensionality(shape)

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
                decompressed_samples = self.decompressed_samples
                new_samples = decompressed_samples + [incoming_sample]  # type: ignore

                compressed_bytes = compress_multiple(
                    new_samples,  # type: ignore
                    compression=self.compression,
                )
                num_compressed_bytes = len(compressed_bytes)
                tiling_threshold = self.tiling_threshold
                if num_compressed_bytes > self.min_chunk_size and not (
                    not decompressed_samples
                    and (
                        tiling_threshold < 0 or num_compressed_bytes > tiling_threshold
                    )
                ):
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
            self.register_in_meta_and_headers(
                None, shape, update_tensor_meta=update_tensor_meta
            )
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
        is_tile: bool = False,
    ):
        if not decompress:
            raise NotImplementedError(
                "`decompress=False` is not supported by chunk compressed chunks as it can cause recompression."
            )
        if self.is_empty_tensor:
            raise EmptyTensorError(
                "This tensor has only been populated with empty samples. "
                "Need to add at least one non-empty sample before retrieving data."
            )
        partial_sample_tile = self._get_partial_sample_tile(as_bytes=False)
        if partial_sample_tile is not None:
            return partial_sample_tile
        if self.is_image_compression:
            return self.decompressed_samples[local_index]  # type: ignore

        decompressed = memoryview(self.decompressed_bytes)  # type: ignore
        if not is_tile and self.is_fixed_shape:
            shape = tuple(self.tensor_meta.min_shape)
            sb, eb = self.get_byte_positions(local_index)
            decompressed = decompressed[sb:eb]
        else:
            shape = self.shapes_encoder[local_index]
            if not self.byte_positions_encoder.is_empty():
                sb, eb = self.byte_positions_encoder[local_index]
                decompressed = decompressed[sb:eb]
        if self.is_text_like:
            return bytes_to_text(decompressed, self.htype)
        ret = np.frombuffer(decompressed, dtype=self.dtype).reshape(shape)
        if copy and not ret.flags["WRITEABLE"]:
            ret = ret.copy()
        return ret

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
        self.check_shape_for_update(shape)
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
        if new_sample is None:
            if self.tensor_meta.max_shape:
                new_sample = np.ones(
                    (0,) * len(self.tensor_meta.max_shape), dtype=self.dtype
                )
            else:
                # earlier sample was also None, do nothing
                return

        new_sample = intelligent_cast(new_sample, self.dtype, self.htype)
        shape = new_sample.shape
        shape = self.normalize_shape(shape)
        if self.is_empty_tensor and len(shape) != 3:
            self.change_dimensionality(shape)
        self.check_shape_for_update(shape)
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
        if sample is None:
            if self.tensor_meta.max_shape:
                shape = (0,) * len(self.tensor_meta.max_shape)
            else:
                # we assume 3d, later we reset dimensions if the assumption was wrong
                shape = (0, 0, 0)
            return np.ones(shape, dtype=self.dtype), None
        if isinstance(sample, SampleTiles):
            return sample, sample.tile_shape
        elif isinstance(sample, PartialSample):
            return (
                SampleTiles(
                    compression=self.compression,
                    chunk_size=self.min_chunk_size,
                    htype=self.htype,
                    sample_shape=sample.sample_shape,
                    tile_shape=sample.tile_shape,
                    dtype=sample.dtype,
                ),
                sample.sample_shape,
            )
        if isinstance(sample, hub.core.tensor.Tensor):
            sample = sample.numpy()
        sample = intelligent_cast(sample, self.dtype, self.htype)
        shape = sample.shape
        shape = self.normalize_shape(shape)
        if not self.is_empty_tensor:
            self.num_dims = self.num_dims or len(shape)
            check_sample_shape(shape, self.num_dims)

        ratio = get_compression_ratio(self.compression)
        approx_compressed_size = sample.nbytes * ratio

        if (
            self.tiling_threshold >= 0
            and approx_compressed_size > self.tiling_threshold
        ):
            sample = SampleTiles(
                sample,
                self.compression,
                self.tiling_threshold,
                store_uncompressed_tiles=True,
            )

        return sample, shape

    def pop(self, index):
        self.prepare_for_write()
        if self.is_byte_compression:
            sb, eb = self.byte_positions_encoder[index]
            self.decompressed_bytes = (
                self.decompressed_bytes[:sb] + self.decompressed_bytes[eb:]
            )
            self._data_bytes = compress_bytes(self.decompressed_bytes, self.compression)
        else:
            self.decompressed_samples.pop(index)
            self._data_bytes = compress_multiple(
                self.decompressed_samples, self.compression
            )
        if not self.shapes_encoder.is_empty():
            self.shapes_encoder.pop(index)
        if not self.byte_positions_encoder.is_empty():
            self.byte_positions_encoder.pop(index)
        self._changed = True

    def pop_multiple(self, num_samples):
        if self.is_byte_compression:
            total_samples = self.num_samples
            self.decompressed_bytes = self.decompressed_bytes[
                : self.byte_positions_encoder[total_samples - num_samples][0]
            ]
            self._data_bytes = compress_bytes(self.decompressed_bytes, self.compression)
        else:
            for _ in range(num_samples):
                self.decompressed_samples.pop()
            self._data_bytes = compress_multiple(
                self.decompressed_samples, self.compression
            )

        for _ in range(num_samples):
            if not self.shapes_encoder.is_empty():
                self.shapes_encoder.pop()
            if not self.byte_positions_encoder.is_empty():
                self.byte_positions_encoder.pop()
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

    @property
    def is_empty_tensor(self):
        if self.is_byte_compression:
            return super().is_empty_tensor
        return self.tensor_meta.max_shape == [0, 0, 0]

    def change_dimensionality(self, shape):
        if len(shape) != 2:
            raise ValueError(
                f"Only amples with shape (H, W) and (H, W, C) are supported in chunks with image compression, got {shape} instead."
            )
        self.tensor_meta.max_shape = list(shape)
        self.tensor_meta.min_shape = list(shape)
        self.num_dims = len(shape)
        empty_shape = (0,) * self.num_dims
        self.tensor_meta.update_shape_interval(empty_shape)
        self.tensor_meta.is_dirty = True
        num_samples = self.shapes_encoder.num_samples
        self.shapes_encoder = ShapeEncoder()
        self.shapes_encoder.register_samples((0,) * len(shape), num_samples)
        if self.decompressed_samples:
            for i, arr in enumerate(self.decompressed_samples):
                self.decompressed_samples[i] = arr.reshape((0,) * self.num_dims)
