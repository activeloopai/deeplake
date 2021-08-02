from hub.core.storage.lru_cache import LRUCache
from typing import Any, Dict, Optional, Union, Sequence
from hub.core.storage.cachable import CachableCallback, use_callback


class Hashlist(CachableCallback):
    def __init__(self):
        """Contains list of hashes generated for a particular tensor."""
        self.hashes = []
        super().__init__()

    def append(self, hash):
        self.hashes.append(hash)

    @property
    def nbytes(self):
        return len(self.tobytes())

    def is_empty(self):
        if len(self.hashes):
            return False
        else:
            return True


def load_hashlist(hashlist_key: str, cache: LRUCache):
    if hashlist_key in cache:
        hashlist = cache.get_cachable(hashlist_key, Hashlist)
    else:
        hashlist = Hashlist()
        hashlist.initialize_callback_location(hashlist_key, cache)

    return hashlist
