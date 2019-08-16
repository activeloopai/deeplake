import hub

def test_mnist():
    x = hub.load('mnist/mnist:train')
    print(x[59995].mean())

def test_imagenet():
    x = hub.load('imagenet')
    print(x[1000000].mean())

if __name__ == "__main__":

    print('Running Dataset Tests')
    test_mnist()
    test_imagenet()