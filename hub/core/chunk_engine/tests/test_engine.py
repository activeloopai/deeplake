import numpy as np

from hub.core.chunk_engine import read, chunk_and_write_array, MemoryProvider
from hub.core.chunk_engine.util import normalize_and_batchify_shape

import pytest

dummy_compressor = lambda x: x
dummy_decompressor = lambda x: x

chunk_size = 4096


# shape,batched
FIXED_SIZE = (
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

    # TODO: make sure there is no more than 1 incomplete chunk at a time. because incomplete chunks are NOT compressed, if there is
    # more than 1 per tensor it can get inefficient

    a_out = read("tensor", dummy_decompressor, storage, cache_chain)

    # writing implicitly normalizes/batchifies shape
    a_in = normalize_and_batchify_shape(a_in, batched=batched)
    print(a_in.shape, a_out.shape, batched)
    np.testing.assert_array_equal(a_in, a_out)


@pytest.mark.parametrize("shape,batched", FIXED_SIZE)
def test_with_cache_chain(shape, batched):
    storage = MemoryProvider()
    cache_chain = [MemoryProvider()]

    a_in = np.random.uniform(size=shape)
    run_test(storage, a_in, batched=batched, cache_chain=cache_chain)
