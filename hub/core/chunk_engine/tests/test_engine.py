import numpy as np
from hub.core.chunk_engine import read, write, MemoryProvider

dummy_compressor = lambda x: x[: len(x) // 2]
chunk_size = 10


def test_example():
    storage = MemoryProvider()
    a = np.arange(100)
    write("tensor", a, storage, dummy_compressor, chunk_size)
