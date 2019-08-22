import hub


def test_mnist():
    x = hub.load('mnist/mnist:train')
    print(x[59995].mean())


def test_imagenet():
    x = hub.load('imagenet')
    print(x[1000000].mean())


def test_dataset():
    x = hub.array((10000, 512, 512), name='test/example:input', dtype='uint8')
    y = hub.array((10000, 4), name='test/example:label', dtype='uint8')
    
    ds = hub.dataset({
        'input': x,
        'label': y
    }, name='test/dataset:train')

    print(ds['input']) # return single array
    print(ds['label', 0]) # equivalent ds['train'][0]
    print(ds[0]) # return pair of arrays as long as dimensions requested are broadcastable
    
def test_load_dataset():
    ds = hub.dataset(name='test/dataset:train') # return the dataset object 
    print(ds.chunk_shape) # returns a dict of shapes
    print(ds.shape)

if __name__ == "__main__":
    print('Running Dataset Tests')
    #test_mnist()
    #test_imagenet()
    #test_dataset()
    test_load_dataset()