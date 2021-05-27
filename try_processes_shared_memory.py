from hub.core.storage import S3Provider, MemoryProvider
from pathos.pools import ProcessPool, ThreadPool
from time import time, sleep
from functools import lru_cache
import mmap
import shutil
import os
from tqdm import tqdm
from multiprocessing import shared_memory, resource_tracker
import array

# from multiprocessing import value, array
@lru_cache()
def s3_client():
    return S3Provider("s3://snark-test/abc-large-3/image/chunks")


def read(file, name):
    remove_shm_from_resource_tracker()
    s3 = s3_client()
    # start = time()
    a = s3[file]
    # FILENAME = f"download/{file}"
    shm = shared_memory.SharedMemory(name=name)
    buffer = shm.buf
    buffer[:] = a
    del buffer
    # shm.close()
    # end = time()
    # print("read took", end-start)
    return 0


# def unreg(name):
#     remove_shm_from_resource_tracker()
#     shm = shared_memory.SharedMemory(name=name)
#     shm.close()
# shm.unlink()
# resource_tracker.unregister(name, 'shared_memory')


def remove_shm_from_resource_tracker():
    """Monkey-patch multiprocessing.resource_tracker so SharedMemory won't be tracked

    More details at: https://bugs.python.org/issue38119
    """

    def fix_register(name, rtype):
        if rtype == "shared_memory":
            return
        return resource_tracker._resource_tracker.register(self, name, rtype)

    resource_tracker.register = fix_register

    def fix_unregister(name, rtype):
        if rtype == "shared_memory":
            return
        return resource_tracker._resource_tracker.unregister(self, name, rtype)

    resource_tracker.unregister = fix_unregister

    if "shared_memory" in resource_tracker._CLEANUP_FUNCS:
        del resource_tracker._CLEANUP_FUNCS["shared_memory"]


workers = 108

thread_pool = ThreadPool(nodes=workers)
process_pool = ProcessPool(nodes=workers)
s3 = S3Provider("s3://snark-test/abc-large-3/image/chunks")
files_to_download = s3._list_keys()


istart = time()
# workers = 128
names = [f"file_{i}" for i in range(workers)]
shms = [
    shared_memory.SharedMemory(create=True, size=16_000_000, name=name)
    for name in names
]
for i in tqdm(range(len(files_to_download) // workers + 1)):

    # results =[array('i', 16*1000*1000)]*workers
    arr = files_to_download[i * workers : (i + 1) * workers]
    start = time()
    results = process_pool.map(read, arr, names, chunksize=1)
    end = time()
    # print(i, "read took", end-start)

    for shm in shms:
        x = shm.buf[:]
        # x.tobytes()
        # print(x[0:100].tobytes())
        del x

# process_pool.map(unreg, names, chunksize=1)
end = time()

print(workers, end - istart)

# for name in names:
#     resource_tracker.unregister(name, 'shared_memory')

for shm in shms:
    shm.close()
    shm.unlink()


# with open(f"download/{files_to_download[0]}", mode="r", encoding="utf8") as file_obj:
#     with mmap.mmap(file_obj.fileno(), length=0, access=mmap.ACCESS_READ) as mmap_obj:
#         text = mmap_obj.read()
#         print(len(text))
# print(text)
