import pytest
from pathos.pools import ProcessPool

from ee.backend.counter import Counter
from ee.backend.utils import redis_loaded


@pytest.mark.skipif(
    not redis_loaded(),
    reason="requires redis to be loaded",
)
def test_counter():

    a = Counter("key2")
    a.reset()
    a.append(10)
    assert a.get() == 10

    pool = ProcessPool(nodes=4)
    samples = [1, 2, 3, 4]

    def store(index):
        Counter("key2").append(10)

    pool.map(store, samples)
    Counter("key2").append(-10)
    assert a.get() == 40


if __name__ == "__main__":
    test_counter()