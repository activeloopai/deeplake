from hub.core.caching.lru_cache import LRUCache  # type: ignore
from hub.core.storage import MemoryProvider, LocalProvider
from hub.util.cache_chain import get_cache_chain

# TODO proper tests, pytest-benchmarks

l1 = MemoryProvider("./abc")
l2 = LocalProvider("./cache/l2")
l3 = LocalProvider("./cache/l3")

lru = get_cache_chain([l1, l2, l3], [64, 100])

lru["file1"] = b"1" * 80

print(lru["file1"])

lru["file2"] = b"5" * 20

print(lru["file1"])

lru["file3"] = b"6" * 30

print(lru["file3"])
print(lru["file1"])
print(lru["file2"])
