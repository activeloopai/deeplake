import hub


def test_mnist():
    print('test mnist')
    x = hub.load('mnist/mnist:train')
    assert x[59995].mean() == 32.910714285714285
    print('passed')


def test_imagenet():
    print('test imagenet')
    x = hub.load('imagenet')
    assert x[1000000].mean() == 163.95653688888888
    print('passed')


def test_dataset():
    x = hub.array((10000, 512, 512), name='test/example:input', dtype='uint8')
    y = hub.array((10000, 4), name='test/example:label', dtype='uint8')

    ds = hub.dataset({
        'input': x,
        'label': y
    }, name='test/dataset:train3')

    assert ds['input'].shape[0] == 10000   # return single array
    assert ds['label', 0].mean() == 0  # equivalent ds['train'][0]
    # return pair of arrays as long as dimensions requested are broadcastable
    assert ds[0][0].mean() == 0


def test_load_dataset():
    ds = hub.dataset(name='test/dataset:train3')  # return the dataset object
    assert ds.chunk_shape == [1]  # returns a dict of shapes
    assert ds.shape == [10000]


if __name__ == "__main__":
    print('Running Dataset Tests')
    test_mnist()
    test_imagenet()
    test_dataset()
    test_load_dataset()
