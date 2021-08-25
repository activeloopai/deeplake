import random
from collections import defaultdict
from typing import Callable, Dict, Optional, Sequence, Set
from hub.core.storage import StorageProvider, SharedMemoryProvider
from hub.core.storage.prefetch_lru_cache import PrefetchLRUCache
from hub.constants import INTELLIGENT_SHUFFLING_THRESHOLD


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
        transform: Optional[Callable],
        mode: Optional[str] = None,
    ):
        super().__init__(
            cache_storage,
            next_storage,
            cache_size,
            dataset,
            num_workers,
            tensor_keys,
            transform,
            mode,
        )

        # set of all indexes that have not been used yet, used to pick new indexes every time
        self.all_remaining_indexes = set(self.all_indexes)

        # keeps count of how many unique tensors have this index in cache, updated in pop and insert
        self.index_ct: Dict[int, int] = defaultdict(int)
        # corresponding to each count, stores the indexes that have appeared that many times
        self.ct_indexes: Dict[int, Set[int]] = defaultdict(set)

        # stores the start and end index of each chunk for each tensor
        self.all_chunks_start_end_index = self._get_all_chunks_start_end_index()

    def remove_index(self, index: int):
        """Removes an index from all the class data structures after it has been used."""
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
        self.all_remaining_indexes = set(self.all_indexes)

    def _suggest_next_index(self) -> int:
        """Suggests the next index to return data from. For shuffle cache this is done by a combination of random picking as well as greedy picking depending on the number of chunks present in the cache for the indexes."""
        if (
            self.cache_used < INTELLIGENT_SHUFFLING_THRESHOLD * self.cache_size
            or not self.index_ct
        ):
            index = random.choice(list(self.all_remaining_indexes))
        else:
            largest_ct = max(self.ct_indexes.keys())
            index = random.choice(list(self.ct_indexes[largest_ct]))
        self.remove_index(index)
        return index

    def _update_count_dicts_insertion(self, tensor, chunk_name):
        """Updates index_ct and ct_index after a new chunk is brought into shared memory."""
        start_index, end_index = self.all_chunks_start_end_index[tensor][chunk_name]
        for index in range(start_index, end_index + 1):
            # TODO: logic will need to be changed once we support big samples that go across chunks
            if index in self.all_remaining_indexes:
                if index in self.index_ct:
                    self.ct_indexes[self.index_ct[index]].discard(index)
                    if len(self.ct_indexes[self.index_ct[index]]) == 0:
                        self.ct_indexes.pop(self.index_ct[index])
                self.index_ct[index] += 1
                self.ct_indexes[self.index_ct[index]].add(index)

    def _update_count_dicts_pop(self, tensor, chunk_name):
        """Updates index_ct and ct_index after a chunk is popped from the cache."""
        start_index, end_index = self.all_chunks_start_end_index[tensor][chunk_name]
        for index in range(start_index, end_index + 1):
            # TODO: logic will need to be changed once we support big samples that go across chunks
            if index in self.all_remaining_indexes:
                self.ct_indexes[self.index_ct[index]].discard(index)
                if len(self.ct_indexes[self.index_ct[index]]) == 0:
                    self.ct_indexes.pop(self.index_ct[index])
                self.index_ct[index] -= 1
                if self.index_ct[index] == 0:
                    self.index_ct.pop(index)
                else:
                    self.ct_indexes[self.index_ct[index]].add(index)
