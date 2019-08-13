import pytest
import hub
import numpy as np

def test_init():
    print('- Initialize array')
    shape = (10,10,10,10)
    x = hub.array(shape, name="test/example:1", dtype='uint8')
    shape = np.array(shape)
    assert np.all(np.array(x.shape) == shape)
    print('passed')

def test_simple_upload_download():
    print('- Simple Chunk Upload and Download')
    x = hub.array((10,10,10,10), name="test/example:1", dtype='uint8')
    x[0] = np.ones((1,10,10,10), dtype='uint8')
    assert x[0].mean() == 1
    print('passed')

def test_multiple_upload_download():
    print('- Multiple Chunk Upload and Download')
    x = hub.array((10,10,10,10), name="test/example:1", dtype='uint8')
    x[0:2] = np.ones((3,10,10,10), dtype='uint8')
    assert x[0:2].mean() == 1
    print('passed')

def test_cross_chunk_upload_download():
    print('- Cross Chunk Upload and Download')
    x = hub.array((100,100,100), name="test/example:1", dtype='uint8')
    x[2:5, 0:10, 0:10] = np.ones((3,10,10), dtype='uint8')
    assert x[2:5, 0:10, 0:10].mean() == 1
    assert x[2:5, 10:, 10:].mean() == 0
    print('passed')


if __name__ == "__main__":
    print('Running Basic Tests')
    test_init()
    test_simple_upload_download()
    test_multiple_upload_download()
    test_cross_chunk_upload_download()
