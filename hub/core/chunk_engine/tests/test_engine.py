import numpy as np
from hub.core.chunk_engine import read, write, MemoryProvider

dummy_compressor = lambda x: x
dummy_decompressor = lambda x: x

chunk_size = 10


def run_test(storage, a_in, cache_chain=[]):
    write(
        "tensor",
        a_in,
        dummy_compressor,
        chunk_size,
        storage,
        cache_chain,
        batched=False,
    )
    a_out = read("tensor", 0, dummy_decompressor, storage, cache_chain)
    np.testing.assert_array_equal(a_in, a_out)


def test_no_cache():
    storage = MemoryProvider()
    a_in = np.random.uniform(size=(100))

    run_test(storage, a_in)


def test_with_cache_chain():
    storage = MemoryProvider()
    cache_chain = [MemoryProvider()]

    a_in = np.random.uniform(size=(100))
    run_test(storage, a_in, cache_chain=cache_chain)
