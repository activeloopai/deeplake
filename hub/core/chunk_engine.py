from typing import Sequence, Union
from hub.util.exceptions import DynamicTensorNumpyError
from hub.core.storage.cachable import Cachable
from hub.core.meta.tensor_meta import TensorMeta
from hub.core.index.index import Index
from hub.util.keys import (
    get_chunk_key,
    get_encoded_chunk_names_key,
    get_tensor_meta_key,
)
import numpy as np

from hub.core.storage.lru_cache import LRUCache

from hub.core.chunk import Chunk

from hub.core.meta.encode.chunk_name import ChunkNameEncoder


SampleValue = Union[np.ndarray, int, float, bool]


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

    def _extend_array(self, array: np.array):
        if len(array.shape) < 1:
            raise ValueError(
                f"Extending requires arrays to have a minimum dimensionality of 1 (`len(shape)`). Got {len(array.shape)}."
            )

        sample_dtype = array.dtype
        num_samples = array.shape[0]
        sample_shape = array.shape[1:]

        self.tensor_meta.check_compatibility(sample_shape, sample_dtype)
        # update tensor meta first because erroneous meta information is better than un-accounted for data.
        self.tensor_meta.update(sample_shape, sample_dtype, num_samples)

        buffer = memoryview(array.tobytes())

        # TODO: we don't always want to create a new chunk (self.last_chunk)

        chunk = self.last_chunk
        if chunk is None:
            chunk = self._create_root_chunk()

        chunk_num_samples_before_extend = chunk.num_samples
        child_chunks = chunk.extend(buffer, num_samples, sample_shape)
        chunk_num_samples_after_extend = chunk.num_samples

        num_new_samples_for_last_chunk = (
            chunk_num_samples_after_extend - chunk_num_samples_before_extend
        )
        self.register_new_samples_for_last_chunk(num_new_samples_for_last_chunk)
        self.register_new_chunks(*child_chunks)

    def extend(self, samples: Sequence[SampleValue]):
        if isinstance(samples, np.ndarray):
            self._extend_array(samples)
        elif isinstance(samples, Sequence):
            try:
                array = np.array(samples)
                self._extend_array(array)
            except:
                for sample in samples:
                    self.append(sample)
        else:
            raise TypeError(f"Unsupported type for extending. Got: {type(samples)}")

    def append(self, sample: SampleValue):
        array = np.array(sample)
        self.extend(np.expand_dims(array, axis=0))

    def _create_root_chunk(self):
        if self.last_chunk is not None:
            raise Exception("You cannot create a root chunk when one already exists.")

        chunk = Chunk()
        self._staged_root_chunk = chunk
        return chunk

    def register_new_samples_for_last_chunk(self, num_new_samples_for_last_chunk: int):
        if num_new_samples_for_last_chunk == 0:
            return

        if self._staged_root_chunk is not None:
            chunk = self._staged_root_chunk
            self.register_new_chunks(chunk)
            self._staged_root_chunk = None

        else:
            chunk = self.last_chunk
            connected_to_next = chunk.next_chunk is not None
            self.index_chunk_name_encoder.attach_samples_to_last_chunk(
                num_new_samples_for_last_chunk, connected_to_next=connected_to_next
            )

    def register_new_chunks(self, *chunks: Chunk):
        for chunk in chunks:
            connected_to_next = chunk.next_chunk is not None

            self.index_chunk_name_encoder.attach_samples_to_new_chunk(
                chunk.num_samples, connected_to_next=connected_to_next
            )

            chunk_name = self.index_chunk_name_encoder.last_chunk_name
            chunk_key = self.get_chunk_key(chunk_name)
            self.cache[chunk_key] = chunk

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
            sample = chunk.get_sample(local_sample_index, self.tensor_meta.dtype)

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
