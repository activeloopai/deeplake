import pytest
import hub
import numpy as np


def test_fs():
    print('- Test writing local to filesystem')
    shape = (10, 10, 10, 10)
    x = hub.array(shape, name="test/example:1", dtype='uint8', backend='fs')
    shape = np.array(shape)
    assert np.all(np.array(x.shape) == shape)
    assert x[0, 0, 0, 0] == 0
    x[1] = 1
    assert x[1, 0, 0, 0] == 1
    print('passed')


if __name__ == "__main__":
    test_fs()
