from typing import Iterator

from numpy.testing._private.utils import assert_array_equal, assert_equal
from hub.core.io import (
    IOBlock,
    Streaming,
    Schedule,
    BufferedStreaming,
    SequentialMultithreadScheduler,
)

import random


class MockStreaming(Streaming):
    def __init__(self) -> None:
        super().__init__()

    def read(self, schedule: Schedule) -> Iterator:
        yield from range(0, 100)


def test_shuffle_buffer():
    random.seed(42)

    under_test = BufferedStreaming(MockStreaming(), 10)

    results = list(under_test.read(None))

    assert_array_equal(results[:5], [1, 0, 6, 5, 7])
    assert_equal(len(results), 100)


def test_sequential_scheduler():
    under_test = SequentialMultithreadScheduler(num_workers=4)

    list_blocks = [IOBlock([], list(range(1, 11))), IOBlock([], [11, 12])]

    result = under_test.schedule(list_blocks)

    assert_array_equal([b.indices() for b in result[0]._blocks], [[1, 5, 9]])
    assert_array_equal([b.indices() for b in result[1]._blocks], [[2, 6, 10]])
    assert_array_equal([b.indices() for b in result[2]._blocks], [[3, 7], [11]])
    assert_array_equal([b.indices() for b in result[3]._blocks], [[4, 8], [12]])
