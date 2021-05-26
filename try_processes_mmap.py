from hub.core.storage import S3Provider, MemoryProvider
from pathos.pools import ProcessPool, ThreadPool
from time import time
from functools import lru_cache
import mmap
import shutil
import os
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
    FILENAME = f"download/{file}"
    f = open(FILENAME, "wb")
    f.write(len(a)*b'\0')
    f.close()
    with open(FILENAME, mode="r+", encoding="utf8") as file_obj:
        with mmap.mmap(file_obj.fileno(), length=0, access=mmap.ACCESS_WRITE) as mmap_obj:
            mmap_obj.write(a)
    # end = time()
    # print("read took", end-start)
    return 0


istart = time()
# workers = 128
for i in range(len(files_to_download)//workers + 1):
    # results =[array('i', 16*1000*1000)]*workers
    arr = files_to_download[i*workers: (i+1)*workers]
    start = time()
    results = process_pool.map(read, arr, chunksize=1)
    end = time()
    print(i, "read took", end-start)

    for a in arr:
        with open(f"download/{a}", mode="r", encoding="utf8") as file_obj:
            with mmap.mmap(file_obj.fileno(), length=0, access=mmap.ACCESS_READ) as mmap_obj:
                text = mmap_obj.read()
                # print(len(text), text[0:10])

    shutil.rmtree("download")
    os.mkdir("download")


# with open(f"download/{files_to_download[0]}", mode="r", encoding="utf8") as file_obj:
#     with mmap.mmap(file_obj.fileno(), length=0, access=mmap.ACCESS_READ) as mmap_obj:
#         text = mmap_obj.read()
#         print(len(text))
        # print(text)
end = time()

print(workers, end-istart)