import numpy as np
from deeplake.util import split


def test_split(memory_ds):
    ds = memory_ds
    ds.create_tensor("ints", dtype="int64")
    ds.ints.extend(np.arange(13, dtype="int64").reshape((13, 1)))

    train, test, val = split(ds, [0.7, 0.2, 0.1])

    expected_train = [[i] for i in range(9)]
    expected_test = [[i + 9] for i in range(2)]
    expected_val = [[i + 11] for i in range(2)]

    assert train.ints.numpy().tolist() == expected_train
    assert test.ints.numpy().tolist() == expected_test
    assert val.ints.numpy().tolist() == expected_val

    assert len(train) == 9
    assert len(test) == 2
    assert len(val) == 2
    assert sum(map(len, (train, test, val))) == 13
