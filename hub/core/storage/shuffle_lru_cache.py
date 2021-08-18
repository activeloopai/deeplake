import random
from collections import defaultdict
from typing import Callable, Optional, Sequence
from hub.core.storage import StorageProvider, SharedMemoryProvider, PrefetchLRUCache


class ShuffleLRUCache(PrefetchLRUCache):
    """Creates an intelligent cache that suggests indexes on the basis of existing cache contents."""

    def __init__(
        self,
        cache_storage: SharedMemoryProvider,
        next_storage: Optional[StorageProvider],
        cache_size: int,
        dataset,
        num_workers: int,
        tensor_keys: Optional[Sequence[str]],
        transform: Callable,
    ):
        super().__init__(
            cache_storage,
            next_storage,
            cache_size,
            dataset,
            num_workers,
            tensor_keys,
            transform,
        )

        self.all_remaining_indexes = set(self.all_indexes)

        # keeps count of how many unique tensors have this index in cache, updated in pop and insert
        self.index_ct = defaultdict(int)
        # corresponding to each count, stores the indexes that have appeared that many times
        self.ct_indexes = defaultdict(set)

        self.all_chunks_start_end_index = self._get_all_chunks_start_end_index()

    def iterate_samples(self, yield_index=False):
        for index, data in super().iterate_samples(yield_index=True):
            self.remove_index(index)
            if yield_index:
                yield index, data
            else:
                yield data

    def remove_index(self, index: int):
        self.all_remaining_indexes.discard(index)
        self.ct_indexes[self.index_ct[index]].discard(index)
        if len(self.ct_indexes[self.index_ct[index]]) == 0:
            self.ct_indexes.pop(self.index_ct[index])
        self.index_ct.pop(index)

    def clear_cache(self):
        """Flushes the content of all the cache layers if not in read mode and and then deletes contents of all the layers of it.
        This doesn't delete data from the actual storage.
        """
        super().clear_cache()
        self.index_ct.clear()
        self.ct_indexes.clear()

    def _suggest_next_index(self) -> int:
        if self.cache_used < 0.8 * self.cache_size or not self.index_ct:
            return random.choice(list(self.all_remaining_indexes))
        largest_ct = max(self.ct_indexes.keys())
        return random.choice(list(self.ct_indexes[largest_ct]))

    def _update_count_dicts_insertion(self, tensor, chunk_name):
        """Updates index_ct and ct_index after a new chunk is brought into shared memory"""
        start_index, end_index = self.all_chunks_start_end_index[tensor][chunk_name]
        for index in range(start_index, end_index + 1):
            # logic will need to be changed once we support big samples that go across chunks
            if index in self.index_ct:
                self.ct_indexes[self.index_ct[index]].discard(index)
                if len(self.ct_indexes[self.index_ct[index]]) == 0:
                    self.ct_indexes.pop(self.index_ct[index])
            self.index_ct[index] += 1
            self.ct_indexes[self.index_ct[index]].add(index)

    def _update_count_dicts_pop(self, tensor, chunk_name):
        start_index, end_index = self.all_chunks_start_end_index[tensor][chunk_name]
        for index in range(start_index, end_index + 1):
            # logic will need to be changed once we support big samples that go across chunks
            self.ct_indexes[self.index_ct[index]].discard(index)
            if len(self.ct_indexes[self.index_ct[index]]) == 0:
                self.ct_indexes.pop(self.index_ct[index])
            self.index_ct[index] -= 1
            if self.index_ct[index] == 0:
                self.index_ct.pop(index)
            else:
                self.ct_indexes[self.index_ct[index]].add(index)
