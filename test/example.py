import hub
import numpy as np

def download():
    vol = hub.load(name='imagenet/image:val')[400:600]
    a = (vol.mean(axis=(1,2,3)) == 0).sum()
    print(vol.mean(axis=(1,2,3)) == 0)

mnist = hub.array((50000, 28, 28, 1), name="jason/mnist:v2", dtype='float32')
mnist[0, :] = np.random.random((1, 28, 28, 1)).astype('float32')

print(mnist[0,0,0,0])
# TODO load
mnist = hub.load(name='jason/mnist:v1')

print(mnist[0].shape)
print(mnist[0,0,0,0])



