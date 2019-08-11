#import gevent.monkey
#gevent.monkey.patch_all(thread=True)

import hub
import time
import numpy as np


x = hub.load('imagenet/fake:train')

if True:
    t1 = time.time()
    zeros = (x[:100].mean(axis=(1,2,3)) == 0).sum()
    t2 = time.time()
    print(zeros, t2-t1)
exit()
for i in range(100):
    if x[i].mean() == 0:
        print(i, 'zero')