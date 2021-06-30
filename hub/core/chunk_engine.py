from hub.core.meta.tensor_meta import TensorMeta
from hub.core.index.index import Index
from hub.util.keys import get_chunk_key, get_tensor_meta_key
import numpy as np

from hub.core.storage.lru_cache import LRUCache

from hub.core.chunk import Chunk

from hub.core.meta.encode.chunk_name import ChunkNameEncoder


class ChunkEngine:
    def __init__(self, key: str, cache: LRUCache):
        if not isinstance(cache, LRUCache):
            raise ValueError(f"Expected cache to be `LRUCache`. Got '{type(cache)}'.")

        self.key = key
        self.cache = cache

        # TODO: load if it already exists
        self.index_chunk_name_encoder = ChunkNameEncoder()

    @property
    def num_chunks(self):
        return self.index_chunk_name_encoder.num_chunks

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

    def extend(self, array: np.array):
        if len(array.shape) < 2:
            raise ValueError(
                f"Extending requires arrays to have a minimum dimensionality of 2 (`len(shape)`). Got {len(array.shape)}."
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

        num_samples_before_extend = chunk.num_full_samples
        child_chunks = chunk.extend(buffer, num_samples, sample_shape)
        num_samples_after_extend = chunk.num_full_samples

        num_new_samples_for_last_chunk = (
            num_samples_after_extend - num_samples_before_extend
        )
        self.register_new_samples_for_last_chunk(num_new_samples_for_last_chunk)
        self.register_new_chunks(*child_chunks)

    def append(self, array: np.array):
        self.extend(np.expand_dims(array, axis=0))

    def _create_root_chunk(self):
        if self.last_chunk is not None:
            raise Exception("You cannot create a root chunk when one already exists.")

        chunk = Chunk()
        self.register_new_chunks(chunk)
        return chunk

    def register_new_samples_for_last_chunk(self, num_new_samples_for_last_chunk: int):
        if num_new_samples_for_last_chunk == 0:
            return

        chunk = self.last_chunk
        connected_to_next = chunk.next_chunk is not None
        self.index_chunk_name_encoder.attach_samples_to_last_chunk(
            num_new_samples_for_last_chunk, connected_to_next=connected_to_next
        )

    def register_new_chunks(self, *chunks: Chunk):
        for chunk in chunks:
            connected_to_next = chunk.next_chunk is not None

            self.index_chunk_name_encoder.attach_samples_to_new_chunk(
                chunk.num_full_samples, connected_to_next=connected_to_next
            )

            chunk_name = self.index_chunk_name_encoder.last_chunk_name
            chunk_key = self.get_chunk_key(chunk_name)
            self.cache[chunk_key] = chunk

    def get_chunk_key(self, chunk_name: str):
        chunk_key = get_chunk_key(self.key, chunk_name)
        return chunk_key

    def numpy(self, index: Index, aslist: bool = False):
        raise NotImplementedError
