from math import ceil
from typing import Sequence, Union, Tuple
from hub.util.exceptions import DynamicTensorNumpyError
from hub.core.storage.cachable import Cachable
from hub.core.meta.tensor_meta import TensorMeta
from hub.core.index.index import Index
from hub.util.keys import (
    get_chunk_key,
    get_encoded_chunk_names_key,
    get_tensor_meta_key,
)
from hub.core.sample import Sample
from hub.constants import UNCOMPRESSED

import numpy as np

from hub.core.storage.lru_cache import LRUCache

from hub.core.chunk import Chunk

from hub.core.meta.encode.chunk_name import ChunkNameEncoder


SampleValue = Union[np.ndarray, int, float, bool, Sample]


class ChunkEngine(Cachable):
    def __init__(self, key: str, cache: LRUCache):
        if not isinstance(cache, LRUCache):
            raise ValueError(f"Expected cache to be `LRUCache`. Got '{type(cache)}'.")

        self.key = key
        self.cache = cache
        self._staged_root_chunk = None

    @property
    def index_chunk_name_encoder(self):
        key = get_encoded_chunk_names_key(self.key)

        try:
            enc = self.cache.get_cachable(key, ChunkNameEncoder)
            return enc
        except KeyError:
            enc = ChunkNameEncoder()
            self.cache[key] = enc
            return enc

    @property
    def num_chunks(self):
        return self.index_chunk_name_encoder.num_chunks

    @property
    def num_samples(self):
        return self.index_chunk_name_encoder.num_samples

    @property
    def last_chunk(self):
        if self.num_chunks == 0:
            return None

        last_chunk_name = self.index_chunk_name_encoder.last_chunk_name
        last_chunk_key = self.get_chunk_key(last_chunk_name)
        return self.cache.get_cachable(last_chunk_key, Chunk)

    @property
    def tensor_meta(self):
        tensor_meta_key = get_tensor_meta_key(self.key)
        return self.cache.get_cachable(tensor_meta_key, TensorMeta)

    def _chunk_bytes(
        self, incoming_buffer: memoryview, shape: Tuple[int, ...], dtype: np.dtype
    ):

        incoming_num_bytes = len(incoming_buffer)
        _validate_incoming_buffer(incoming_num_bytes, shape)
        num_samples = shape[0]
        sample_shape = shape[1:]

        # update tensor meta first because erroneous meta information is better than un-accounted for data.
        self.tensor_meta.check_compatibility(sample_shape, dtype)
        self.tensor_meta.update(sample_shape, dtype, num_samples)

        parent_chunk = self.last_chunk or Chunk()
        max_data_bytes = parent_chunk.max_data_bytes

        if parent_chunk.is_under_min_space:
            last_chunk_size = parent_chunk.num_data_bytes
            chunk_ct_content = _min_chunk_ct_for_data_size(
                max_data_bytes, incoming_num_bytes
            )

            extra_bytes = min(incoming_num_bytes, max_data_bytes - last_chunk_size)
            combined_chunk_ct = _min_chunk_ct_for_data_size(
                max_data_bytes, incoming_num_bytes + last_chunk_size
            )

            if combined_chunk_ct == chunk_ct_content:  # combine if count is same
                # start_byte = index_meta.entries[-1]["end_byte"]
                # start_byte = parent_chunk.num_data_bytes
                # end_byte = start_byte + extra_bytes
                parent_chunk.extend(incoming_buffer[:extra_bytes])
                forwarding_buffer = incoming_buffer[extra_bytes:]

        children = []
        while len(forwarding_buffer) > 0:
            child_chunk = Chunk(max_data_bytes=max_data_bytes)
            end_byte = min(len(forwarding_buffer), max_data_bytes)

            # end_byte = min(len(content), CHUNK_MAX_SIZE)
            child_chunk.extend(forwarding_buffer[:end_byte])
            forwarding_buffer = forwarding_buffer[end_byte:]

            children.append(child_chunk)

        parent_chunk.update_headers(incoming_num_bytes, num_samples, sample_shape)
        self.register_chunks(parent_chunk, children)

        """
        index_meta.add_entry(
            chunk_names=chunk_names,
            start_byte=start_byte,
            end_byte=end_byte,
            **extra_sample_meta,
        )
        """

    def register_chunks(self, parent_chunk: Chunk, children_chunks: Sequence[Chunk]):
        if parent_chunk == self.last_chunk:
            raise NotImplementedError
        else:
            raise NotImplementedError

        for child_chunk in children_chunks:
            pass

        raise NotImplementedError

    def extend(self, samples: Union[np.ndarray, Sequence[SampleValue]]):
        if isinstance(samples, np.ndarray):
            compression = self.tensor_meta.sample_compression
            if compression == UNCOMPRESSED:
                buffer = memoryview(samples.tobytes())
                self._chunk_bytes(buffer, samples.shape, samples.dtype)
            else:
                for sample in samples:
                    self.append(sample)
        elif isinstance(samples, Sequence):
            if any(isinstance(s, Sample) for s in samples):
                for sample in samples:
                    self.append(sample)
            else:
                try:
                    self.extend(np.array(samples))
                except:
                    for sample in samples:
                        self.append(sample)
        else:
            raise TypeError(f"Unsupported type for extending. Got: {type(samples)}")

    def append(self, sample: SampleValue):
        if isinstance(sample, Sample):
            # has to decompress to read the array's shape and dtype
            # might be able to optimize this away
            shape = (1, *sample.shape)
            compression = self.tensor_meta.sample_compression
            data = memoryview(sample.compressed_bytes(compression))
            self._chunk_bytes(data, shape, sample.dtype)
        else:
            return self.append(Sample(array=np.array(sample)))

    def get_chunk_key(self, chunk_name: str):
        chunk_key = get_chunk_key(self.key, chunk_name)
        return chunk_key

    def numpy(self, index: Index, aslist: bool = False):
        # TODO: get chunks from cache in parallel

        length = self.num_samples
        enc = self.index_chunk_name_encoder
        last_shape = None
        samples = []

        for global_sample_index in index.values[0].indices(length):
            first_chunk_name = enc.get_chunk_names(global_sample_index, first_only=True)

            chunk_key = self.get_chunk_key(first_chunk_name)
            chunk: Chunk = self.cache.get_cachable(chunk_key, Chunk)
            local_sample_index = enc.get_local_sample_index(global_sample_index)
            default_compress = self.tensor_meta.sample_compression != UNCOMPRESSED
            sample = chunk.get_sample(
                local_sample_index,
                self.tensor_meta.dtype,
                expect_compressed=default_compress,
            )

            if not aslist and last_shape is not None:
                if sample.shape != last_shape:
                    raise DynamicTensorNumpyError(self.key, index, "shape")

            last_shape = sample.shape
            samples.append(sample)

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


def _validate_incoming_buffer(
    incoming_num_bytes: bytes,
    shape: Tuple[int],
):
    if len(shape) < 1:
        raise ValueError(
            f"Extending requires arrays to have a minimum dimensionality of 1 (`len(shape)`). Got {len(shape)}."
        )

    num_samples = shape[0]
    if num_samples <= 0:
        raise ValueError(
            f"The number of samples a buffer can represent has to be greater than 0. Got {num_samples}"
        )

    if incoming_num_bytes % num_samples != 0:
        raise ValueError(
            f"Incoming buffer length should be perfectly divisible by the number of samples it represents. length={incoming_num_bytes}, num_samples={num_samples}"
        )
