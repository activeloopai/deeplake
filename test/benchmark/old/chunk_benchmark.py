from hub.marray.array import HubArray
import time
import numpy as np


chunk_sizes = [
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    8192 * 2,
    8192 * 4,
    8192 * 8,
]

download_time = []
upload_time = []
for cs in chunk_sizes:
    x = HubArray(
        (cs, cs),
        key="test/benchmark:t{}".format(str(cs)),
        chunk_shape=(cs, cs),
        dtype="uint8",
        compression=None,
    )
    arr = (255 * np.random.rand(cs, cs)).astype("uint8")

    # Upload
    t1 = time.time()
    x[:] = arr
    t2 = time.time()
    upload_time.append(t2 - t1)

    # Download
    t3 = time.time()
    x[:]
    t4 = time.time()
    download_time.append(t4 - t3)

    print(
        "chunk size {} download in {} and uploaded in {}".format(cs, t4 - t3, t2 - t1)
    )

print(download_time)
print(upload_time)
