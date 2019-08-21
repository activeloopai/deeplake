# Taken from https://github.com/hsjeong5/MNIST-for-Numpy
import hub
import numpy as np
from urllib import request
import gzip
import pickle
import time

filename = [
    ["train", "train-images-idx3-ubyte.gz"],
    ["train_labels", "t10k-images-idx3-ubyte.gz"],
    ["test", "train-labels-idx1-ubyte.gz"],
    ["test_labels", "t10k-labels-idx1-ubyte.gz"]
]


def download_mnist():
    base_url = "http://yann.lecun.com/exdb/mnist/"
    for name in filename:
        print("Downloading "+name[1]+"...")
        request.urlretrieve(base_url+name[1], name[1])
    print("Download complete.")


def save_mnist():
    mnist = {}
    for name in filename[:2]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(
                f.read(), np.uint8, offset=16).reshape(-1, 28*28)
    for name in filename[-2:]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
    with open("mnist.pkl", 'wb') as f:
        pickle.dump(mnist, f)
    print("Save complete.")


def init():
    download_mnist()
    save_mnist()


def load():
    with open("mnist.pkl", 'rb') as f:
        mnist = pickle.load(f)
    return mnist


if __name__ == '__main__':
    init()
    arrays = load()
    chunk_length = 128
    for key in arrays:
        t1 = time.time()
        obj = arrays[key]
        shape = obj.shape
        chunk_size = np.array(shape)
        chunk_size[0] = chunk_length
        x = hub.array(shape, name='mnist/mnist_test:{}'.format(key),
                      chunk_size=chunk_size.tolist(), dtype='uint8')
        x[:] = obj
        t2 = time.time()
        print('uploaded {} {} in {}s'.format(key, shape, t2-t1))
