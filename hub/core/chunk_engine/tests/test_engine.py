import numpy as np

from hub.core.chunk_engine import read, chunk_and_write_array, MemoryProvider
from hub.core.chunk_engine.util import normalize_and_batchify_shape
from hub.core.chunk_engine.dummy_util import (
    MemoryProvider,
    DummySampleCompression,
    DummyChunkCompression,
)

import pytest


chunk_size = 16000000


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


def run_test(a_in, storage, compression, batched):
    chunk_and_write_array(
        a_in,
        "tensor",
        compression,
        chunk_size,
        storage,
        batched=batched,
    )

    # TODO: make sure there is no more than 1 incomplete chunk at a time. because incomplete chunks are NOT compressed, if there is
    # more than 1 per tensor it can get inefficient

    # TODO:
    a_out = read("tensor", storage)

    # writing implicitly normalizes/batchifies shape
    a_in = normalize_and_batchify_shape(a_in, batched=batched)
    # print(a_in.shape, a_out.shape, batched)
    np.testing.assert_array_equal(a_in, a_out)


@pytest.mark.parametrize("shape,batched", FIXED_SIZE)
def test_fixed_size_chunk_compression(shape, batched):
    storage = MemoryProvider()
    compression = DummyChunkCompression()

    a_in = np.random.uniform(size=shape)
    run_test(a_in, storage, compression, batched=batched)


@pytest.mark.parametrize("shape,batched", FIXED_SIZE)
def test_fixed_size_sample_compression(shape, batched):
    storage = MemoryProvider()
    compression = DummySampleCompression()

    a_in = np.random.uniform(size=shape)
    run_test(a_in, storage, compression, batched=batched)
