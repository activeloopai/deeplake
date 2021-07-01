from hub.core.compression import decompress_array
from math import ceil
from typing import Sequence, Union, Tuple
from hub.util.exceptions import DynamicTensorNumpyError
from hub.core.storage.cachable import Cachable
from hub.core.meta.tensor_meta import TensorMeta
from hub.core.index.index import Index
from hub.util.keys import (
    get_chunk_key,
    get_chunk_id_encoder_key,
    get_tensor_meta_key,
)
from hub.core.sample import Sample
from hub.constants import DEFAULT_MAX_CHUNK_SIZE, UNCOMPRESSED

import numpy as np

from hub.core.storage.lru_cache import LRUCache

from hub.core.chunk import Chunk

from hub.core.meta.encode.chunk_id import ChunkIdEncoder


SampleValue = Union[np.ndarray, int, float, bool, Sample]


def is_uniform_sequence(samples):
    if len(set(map(type, samples))) != 1:
        # Cannot vectorize sequence with inconsistent types
        return False
    elif any(isinstance(s, np.ndarray) for s in samples):
        # Numpy arrays will only be vectorized if they have the same shape
        return len(set(s.shape for s in samples)) == 1
    elif any(isinstance(s, Sample) for s in samples):
        # Sample objects will not be vectorized
        return False
    else:
        # Scalar samples can be vectorized
        return True


class ChunkEngine(Cachable):
    def __init__(
        self, key: str, cache: LRUCache, max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE
    ):
        if not isinstance(cache, LRUCache):
            raise ValueError(f"Expected cache to be `LRUCache`. Got '{type(cache)}'.")

        self.key = key
        self.cache = cache

        if max_chunk_size <= 2:
            raise ValueError("Max chunk size should be > 2 bytes.")

        self.max_chunk_size = max_chunk_size
        self.min_chunk_size_target = self.max_chunk_size // 2

    @property
    def chunk_id_encoder(self):
        key = get_chunk_id_encoder_key(self.key)

        try:
            enc = self.cache.get_cachable(key, ChunkIdEncoder)
            return enc
        except KeyError:
            enc = ChunkIdEncoder()
            self.cache[key] = enc
            return enc

    @property
    def num_chunks(self):
        return self.chunk_id_encoder.num_chunks

    @property
    def num_samples(self):
        return self.chunk_id_encoder.num_samples

    @property
    def last_chunk(self):
        if self.num_chunks == 0:
            return None

        last_chunk_name = self.chunk_id_encoder.get_name_for_chunk(-1)
        last_chunk_key = get_chunk_key(self.key, last_chunk_name)
        return self.cache.get_cachable(last_chunk_key, Chunk)

    @property
    def tensor_meta(self):
        tensor_meta_key = get_tensor_meta_key(self.key)
        return self.cache.get_cachable(tensor_meta_key, TensorMeta)

    def _append_bytes(
        self, incoming_buffer: memoryview, shape: Tuple[int, ...], dtype: np.dtype
    ):
        # TODO: docstring

        num_samples = 1
        incoming_num_bytes = len(incoming_buffer)

        # update tensor meta first because erroneous meta information is better than un-accounted for data.
        self.tensor_meta.check_compatibility(shape, dtype)
        self.tensor_meta.update(shape, dtype, num_samples)

        last_chunk = self.last_chunk or self._create_new_chunk()
        max_data_bytes = last_chunk.max_data_bytes
        last_chunk_extended = False

        forwarding_buffer = incoming_buffer
        if last_chunk.is_under_min_space:
            last_chunk_size = last_chunk.num_data_bytes
            chunk_ct_content = _min_chunk_ct_for_data_size(
                max_data_bytes, incoming_num_bytes
            )

            extra_bytes = min(incoming_num_bytes, max_data_bytes - last_chunk_size)
            combined_chunk_ct = _min_chunk_ct_for_data_size(
                max_data_bytes, incoming_num_bytes + last_chunk_size
            )

            # combine if count is same
            if combined_chunk_ct == chunk_ct_content:
                last_chunk.append(forwarding_buffer[:extra_bytes])
                forwarding_buffer = forwarding_buffer[extra_bytes:]
                self._synchronize_chunk(last_chunk, connect_with_last=False)
                last_chunk_extended = True

        new_chunks = []
        connect_with_last = last_chunk_extended

        # `or not connect_with_last` is necessary to support empty samples that weren't written to the previous chunk
        while len(forwarding_buffer) > 0 or not connect_with_last:
            new_chunk = self._create_new_chunk()
            end_byte = min(len(forwarding_buffer), max_data_bytes)

            new_chunk.append(forwarding_buffer[:end_byte])
            forwarding_buffer = forwarding_buffer[end_byte:]

            self._synchronize_chunk(new_chunk, connect_with_last=connect_with_last)

            new_chunks.append(new_chunk)
            connect_with_last = True

        # only the head chunk (the first chunk this sample was written to) should have it's headers updated
        head_chunk = last_chunk if last_chunk_extended else new_chunks[0]
        head_chunk.update_headers(incoming_num_bytes, num_samples, shape)

    def _synchronize_chunk(self, chunk: Chunk, connect_with_last: bool = False):
        # TODO: docstring

        if chunk.num_new_samples <= 0:
            # TODO: exceptions.py
            raise Exception("This chunk has no new samples to be synchronized.")

        num_new_samples = chunk.num_new_samples
        if connect_with_last:
            # if connected with last, there are no new samples, only a continuation of the previous
            num_new_samples = 0
            self.chunk_id_encoder.register_connection()

        self.chunk_id_encoder.register_samples_to_last_chunk_id(num_new_samples)

    def _create_new_chunk(self):
        chunk_id = self.chunk_id_encoder.generate_chunk_id()
        chunk = Chunk(self.max_chunk_size, self.min_chunk_size_target)
        chunk_name = ChunkIdEncoder.name_from_id(chunk_id)
        chunk_key = get_chunk_key(self.key, chunk_name)
        self.cache[chunk_key] = chunk
        return chunk

    def extend(self, samples: Union[np.ndarray, Sequence[SampleValue]]):
        if isinstance(samples, np.ndarray):
            compression = self.tensor_meta.sample_compression
            if compression == UNCOMPRESSED:
                for sample in samples:
                    buffer = memoryview(sample.tobytes())
                    self._append_bytes(buffer, sample.shape, sample.dtype)
            else:
                for sample in samples:
                    self.append(sample)
        elif isinstance(samples, Sequence):
            if is_uniform_sequence(samples):
                self.extend(np.array(samples))
            else:
                for sample in samples:
                    self.append(sample)
        else:
            raise TypeError(f"Unsupported type for extending. Got: {type(samples)}")

    def append(self, sample: SampleValue):
        if isinstance(sample, Sample):
            # has to decompress to read the array's shape and dtype
            # might be able to optimize this away
            compression = self.tensor_meta.sample_compression
            data = memoryview(sample.compressed_bytes(compression))
            self._append_bytes(data, sample.shape, sample.dtype)
        else:
            return self.append(Sample(array=np.array(sample)))

    def numpy(self, index: Index, aslist: bool = False):
        # TODO: get chunks from cache in parallel

        tensor_meta = self.tensor_meta

        expect_compressed = tensor_meta.sample_compression != UNCOMPRESSED
        dtype = tensor_meta.dtype

        length = self.num_samples
        enc = self.chunk_id_encoder
        last_shape = None
        samples = []

        for global_sample_index in index.values[0].indices(length):
            chunk_ids = enc.__getitem__(global_sample_index)

            buffer: Union[bytearray, memoryview] = bytearray()
            for i, chunk_id in enumerate(chunk_ids):
                chunk_name = ChunkIdEncoder.name_from_id(chunk_id)

                chunk_key = get_chunk_key(self.key, chunk_name)
                chunk: Chunk = self.cache.get_cachable(chunk_key, Chunk)
                local_sample_index = enc.get_local_sample_index(global_sample_index)

                # head chunk is the first chunk a sample lives in (has the header information for that sample)
                is_head_chunk = i == 0
                if is_head_chunk:
                    shape = chunk.shape_encoder[local_sample_index]
                    sb, eb = chunk.byte_positions_encoder[local_sample_index]

                else:
                    raise NotImplementedError

                # TODO: optimize this to reduce memory copies for samples spanning accross chunks
                if len(chunk_ids) == 1:
                    # if sample lives in a single chunk, no need to copy the data
                    buffer = chunk.memoryview_data
                else:
                    buffer += chunk.memoryview_data

                if not aslist and last_shape is not None:
                    if shape != last_shape:
                        raise DynamicTensorNumpyError(self.key, index, "shape")

            buffer = buffer[sb:eb]
            if expect_compressed:
                sample = decompress_array(buffer, shape)
            else:
                sample = np.frombuffer(buffer, dtype=dtype).reshape(shape)
            samples.append(sample)

            last_shape = shape

        return _format_samples(samples, index, aslist)


def _format_samples(samples: Sequence[np.array], index: Index, aslist: bool):
    # TODO: docstring

    samples = index.apply(samples)

    if aslist and all(map(np.isscalar, samples)):
        samples = list(arr.item() for arr in samples)

    samples = index.apply_squeeze(samples)

    if aslist:
        return samples
    else:
        return np.array(samples)


def _min_chunk_ct_for_data_size(chunk_max_data_bytes: int, size: int) -> int:
    """Calculates the minimum number of chunks in which data of given size can be fit."""
    return ceil(size / chunk_max_data_bytes)
