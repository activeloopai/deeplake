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

        last_chunk_name = self.index_chunk_name_encoder.get_name_for_chunk(-1)
        return self.cache.get_cachable(last_chunk_name, Chunk)

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

        buffer = array.tobytes()

        # TODO: we don't always want to create a new chunk (self.last_chunk)
        chunk = Chunk()
        extra_chunks = chunk.extend(buffer, num_samples, sample_shape)

        self.register_new_chunks(chunk, *extra_chunks)
        self.tensor_meta.update(sample_shape, sample_dtype, num_samples)

        raise NotImplementedError

    def append(self, array: np.array):
        self.extend(np.expand_dims(array, axis=0))

    def register_new_chunks(self, *chunks: Chunk):
        for chunk in chunks:
            connected_to_next = chunk.next_chunk is not None

            self.index_chunk_name_encoder.attach_samples_to_last_chunk(
                chunk.num_samples, connected_to_next=connected_to_next
            )
            chunk_name = self.index_chunk_name_encoder.last_chunk_name
            chunk_key = get_chunk_key(self.key, chunk_name)

            self.cache[chunk_key] = chunk

    def numpy(self, index: Index, aslist: bool = False):
        raise NotImplementedError
