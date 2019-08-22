import pytest
import hub
import numpy as np


def test_fs():
    print('- Test writing local to filesystem')
    shape = (10, 10, 10, 10)
    x = hub.array(shape, name="test/backend:1", dtype='uint8', backend='fs')
    shape = np.array(shape)
    assert np.all(np.array(x.shape) == shape)
    assert x[0, 0, 0, 0] == 0
    x[1] = 1
    assert x[1, 0, 0, 0] == 1
    print('passed')

def test_s3():
    print('- Test writing local to s3')
    shape = (10, 10, 10, 10)
    x = hub.array(shape, name="test/backend:2", dtype='uint8', backend='s3')
    shape = np.array(shape)
    assert np.all(np.array(x.shape) == shape)
    assert x[0, 0, 0, 0] == 0
    x[1] = 1
    assert x[1, 0, 0, 0] == 1
    print('passed')


def test_multiple():
    print('- Test writing local to filesystem and s3')
    shape = (10, 10, 10, 10)
    x = hub.array(shape, name="test/backend:3", dtype='uint8', backend=['fs', 's3'])
    shape = np.array(shape)
    assert np.all(np.array(x.shape) == shape)
    assert x[0, 0, 0, 0] == 0
    x[1] = 1
    assert x[1, 0, 0, 0] == 1
    print('passed')


def test_cache():
    print('- Test write to cache')
    shape = (10, 10, 10, 10)
    x = hub.array(shape, name="test/backend:4", dtype='uint8', backend=['fs', 's3'], caching=True)
    assert np.all(np.array(x.shape) == shape)
    assert x[0, 0, 0, 0] == 0
    x[1] = 1
    assert x[1, 0, 0, 0] == 1
    print('passed')
    
if __name__ == "__main__":
    test_fs()
    test_s3()
    test_multiple()
