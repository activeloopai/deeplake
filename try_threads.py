from pathos.pools import ProcessPool, ThreadPool
import time
import numpy as np
def fn(a):
    arr = np.random.randint(0, 100, size=(10000, 10000))
    return a

workers = 15

inp = [8] * workers
start = time.time()
ls = ProcessPool(nodes=workers).map(fn ,inp)
end = time.time()
print(workers, end-start)


