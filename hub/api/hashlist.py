from hub.core.storage.lru_cache import LRUCache
from typing import Any, Dict, Optional, Union, Sequence
from hub.core.storage.cachable import CachableCallback, use_callback


class Hashlist(CachableCallback):
    def __init__(self):
        """Contains **optional** key/values that datasets/tensors use for human-readability.
        See the `Meta` class for required key/values for datasets/tensors.

        Note:
            Since `Info` is rarely written to and mostly by the user, every modifier will call `cache[key] = self`.
            Must call `initialize_callback_location` before using any methods.
        """
        self.hashes = []
        super().__init__()

    def append(self, hash):
        self.hashes.append(hash)

    @property
    def nbytes(self):
        # TODO: optimize this
        return len(self.tobytes())

    @property
    def print(self):
        text = "hello"
        print(text)


def load_hashlist(hashlist_key: str, cache: LRUCache):
    if hashlist_key in cache:
        hashlist = cache.get_cachable(hashlist_key, Hashlist)
    else:
        hashlist = Hashlist()
        hashlist.initialize_callback_location(hashlist_key, cache)

    return hashlist

