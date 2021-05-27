from hub.core.storage import S3Provider, MemoryProvider
from pathos.pools import ProcessPool, ThreadPool
from time import time
from functools import lru_cache

# from multiprocessing import value, array
@lru_cache()
def s3_client():
    return S3Provider("s3://snark-test/abc-large-3/image/chunks")


workers = 72

thread_pool = ThreadPool(nodes=workers)
process_pool = ProcessPool(nodes=workers)
s3 = S3Provider("s3://snark-test/abc-large-3/image/chunks")
files_to_download = s3._list_keys()


def read(file):
    s3 = s3_client()
    # start = time()
    a = s3[file]
    # end = time()
    # print("read took", end-start)
    return 0


istart = time()
# workers = 128
for i in range(len(files_to_download) // workers + 1):
    # results =[array('i', 16*1000*1000)]*workers
    arr = files_to_download[i * workers : (i + 1) * workers]
    start = time()
    results = process_pool.map(read, arr, chunksize=1)
    end = time()
    print(i, "read took", end - start)
end = time()

print(workers, end - istart)
