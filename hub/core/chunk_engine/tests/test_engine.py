import numpy as np
from hub.core.chunk_engine import read, chunk_and_write_array, MemoryProvider

import pytest

dummy_compressor = lambda x: x
dummy_decompressor = lambda x: x

chunk_size = 10


# shape,batched
ARRAY_SPECS = (
    # unbatched
    ((1,), False),
    ((100,), False),
    ((100, 100), False),
    ((100, 100, 3), False),
    ((3, 100, 100), False),
    ((3, 100, 100, 1), False),
    # batched
    ((1, 1), True),
    ((10, 1), True),
    ((1, 100), True),
    ((10, 100), True),
    ((1, 100, 100), True),
    ((10, 100, 100), True),
    ((1, 100, 100, 3), True),
    ((10, 100, 100, 3), True),
    ((1, 3, 100, 100), True),
    ((10, 3, 100, 100), True),
    ((1, 3, 100, 100, 1), True),
    ((10, 3, 100, 100, 1), True),
)


def run_test(storage, a_in, batched, cache_chain=[]):
    chunk_and_write_array(
        a_in,
        "tensor",
        dummy_compressor,
        chunk_size,
        storage,
        cache_chain,
        batched=batched,
    )

    for cache in cache_chain:
        assert (
            len(cache.mapper.keys()) == 0
        ), "write_array(...) should implicitly flush the cache (clear & migrate everything to storage)"

    a_out = read("tensor", 0, dummy_decompressor, storage, cache_chain)
    np.testing.assert_array_equal(a_in, a_out)


@pytest.mark.parametrize("shape,batched", ARRAY_SPECS)
def test_no_cache(shape, batched):
    storage = MemoryProvider()
    a_in = np.random.uniform(size=shape)

    run_test(storage, a_in, batched=batched)


@pytest.mark.parametrize("shape,batched", ARRAY_SPECS)
def test_with_cache_chain(shape, batched):
    storage = MemoryProvider()
    cache_chain = [MemoryProvider()]

    a_in = np.random.uniform(size=shape)
    run_test(storage, a_in, batched=batched, cache_chain=cache_chain)
