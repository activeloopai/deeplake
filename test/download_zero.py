#import gevent.monkey
#gevent.monkey.patch_all(thread=True)

import meta
import time
import numpy as np

#x = meta.array((1000,200,200), 'test/image:train2', dtype='uint8')
#x[:990] = (255*np.random.random((1000-10,200,200))).astype('uint8')
#print(x.shape, x.mean())
#exit()
x = meta.load('imagenet/fake:train')

if True:
    t1 = time.time()
    zeros = (x[:100].mean(axis=(1,2,3)) == 0).sum()
    t2 = time.time()
    print(zeros, t2-t1)
exit()
for i in range(100):
    if x[i].mean() == 0:
        print(i, 'zero')