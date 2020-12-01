import time

import numpy as np

from hub.areal.storage_tensor import StorageTensor


def main():
    # Change memcache value between None and 100 to see the difference in performance

    t = StorageTensor(
        "s3://snark-test/benchmarks/areal_caching_benchmark",
        shape=(200, 200, 200),
        memcache=100,
    )
    print(f"Tensor chunks: {t.chunks}")
    t[:, :, :] = 5
    tstart = time.time()
    arr = 5 * np.ones((100, 100))
    for i in range(100, 110):
        arr = t[i, 100:200, 100:200]
    tend = time.time()
    print(f"Total time: {tend - tstart}s")


if __name__ == "__main__":
    main()
