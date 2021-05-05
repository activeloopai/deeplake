import numpy as np
from hub.core.chunk_engine import read, write, MemoryProvider

dummy_compressor = lambda x: x
dummy_decompressor = lambda x: x

chunk_size = 10


def run_test(storage, a_in, cache_chain=[]):
    write(
        "tensor", a_in, storage, dummy_compressor, chunk_size, cache_chain=cache_chain
    )
    a_out = read("tensor", 0, storage, dummy_decompressor)


def test_no_cache():
    storage = MemoryProvider()
    a_in = np.random.uniform((1, 100))

    run_test(storage, a_in)
    print(storage.used_space)
    print(storage.mapper)


def test_with_cache_chain():
    storage = MemoryProvider()
    cache_chain = [MemoryProvider()]

    a_in = np.random.uniform((1, 100))
    run_test(storage, a_in, cache_chain=cache_chain)
