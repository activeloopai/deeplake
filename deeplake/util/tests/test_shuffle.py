import numpy as np
from deeplake.util import shuffle


def test_shuffle(memory_ds):
    ds = memory_ds
    ds.create_tensor("ints", dtype="int64")
    ds.ints.extend(np.arange(10, dtype="int64").reshape((10, 1)))

    np.random.seed(0)
    ds = shuffle(ds)
    expected = [[2], [8], [4], [9], [1], [6], [7], [3], [0], [5]]
    assert ds.ints.numpy().tolist() == expected

    assert ds.ints[0].numpy() == 2
