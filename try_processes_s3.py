from hub.core.storage import S3Provider
from pathos.pools import ProcessPool, ThreadPool
from time import time
from functools import lru_cache

@lru_cache()
def s3_client():
    return S3Provider("s3://snark-test/abc-large-3/image/chunks")

workers = 16

thread_pool = ThreadPool(nodes=workers)
process_pool = ProcessPool(nodes=workers)
s3 = S3Provider("s3://snark-test/abc-large-3/image/chunks")
files_to_download = s3._list_keys()

def read(file):
    s3 = s3_client()
    return s3[file]
start = time()
process_pool.map(read, files_to_download, chunks=1)
end = time()

print(workers, end-start)


