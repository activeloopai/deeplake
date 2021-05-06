import numpy as np
from hub.core.chunk_engine import read, write_array, MemoryProvider

dummy_compressor = lambda x: x
dummy_decompressor = lambda x: x

chunk_size = 10


def run_test(storage, a_in, cache_chain=[]):
    write_array(
        a_in,
        "tensor",
        dummy_compressor,
        chunk_size,
        storage,
        cache_chain,
        batched=False,
    )

    for cache in cache_chain:
        assert (
            len(cache.mapper.keys()) == 0
        ), "write_array(...) should implicitly flush the cache (clear & migrate everything to storage)"

    print(storage.mapper.keys())

    # a_out = read("tensor", 0, dummy_decompressor, storage, cache_chain)
    # np.testing.assert_array_equal(a_in, a_out)


def test_no_cache():
    storage = MemoryProvider()
    a_in = np.random.uniform(size=(100))

    run_test(storage, a_in)


def test_with_cache_chain():
    storage = MemoryProvider()
    cache_chain = [MemoryProvider()]

    a_in = np.random.uniform(size=(100))
    run_test(storage, a_in, cache_chain=cache_chain)
