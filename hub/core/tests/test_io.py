from typing import Iterator

from numpy.testing._private.utils import assert_array_equal, assert_equal
from hub.core.io import Streaming, Schedule, BufferedStreaming

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
