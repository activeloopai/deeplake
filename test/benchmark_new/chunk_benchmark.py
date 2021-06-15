import hub_v1
from hub_v1.schema import Tensor
import time
import numpy as np


chunk_sizes = [
    32,
    64,
    128,
    256,
    512,
    1024,
    # 2048,
    # 4096,
    # 8192,
    # 8192 * 2,
    # 8192 * 4,
    # 8192 * 8,
]

download_time = []
upload_time = []
for cs in chunk_sizes:
    shape = (1,)
    my_schema = {
        "img": Tensor(shape=(cs, cs), chunks=cs, dtype="uint8", compressor="default")
    }
    ds = hub_v1.Dataset(
        "test/benchmark:t{}".format(str(cs)), shape=shape, schema=my_schema
    )
    arr = (255 * np.random.rand(shape[0], cs, cs)).astype("uint8")

    # Upload
    t1 = time.time()
    ds["img"][:] = arr
    t2 = time.time()
    upload_time.append(t2 - t1)

    # Download
    t3 = time.time()
    ds["img"][:]
    t4 = time.time()
    download_time.append(t4 - t3)

    ds.close()
    print(
        "chunk size {} download in {} and uploaded in {}".format(cs, t4 - t3, t2 - t1)
    )

print(download_time)
print(upload_time)
