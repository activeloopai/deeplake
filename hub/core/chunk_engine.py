import numpy as np
from hub.core.storage.lru_cache import LRUCache


class ChunkEngine:
    def __init__(self, key: str, cache: LRUCache):
        if not isinstance(cache, LRUCache):
            raise ValueError(f"Expected cache to be `LRUCache`. Got '{type(cache)}'.")

        self.key = key
        self.cache = cache

        self.index_chunk_encoder = None

    def extend(self, array: np.array):
        raise NotImplementedError

    def append(self, array: np.array):
        raise NotImplementedError
