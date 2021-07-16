from time import time
from shutil import rmtree
import numpy as np
import cProfile
import hub
from tqdm import tqdm


def upload():
    path = f"./mock_mnist"
    rmtree(path, ignore_errors=True)
    ds = hub.Dataset(path)

    # ds = hub.Dataset("s3://internal-datasets/dummy-mnist-NEW-META")

    # ds = hub.Dataset(f"hub://dyllan/dummy-mnist-NEW-META")

    ds.create_tensor("image", htype="image", sample_compression=None)
    ds.create_tensor("label")

    # N = 60000
    N = 5000
    with ds:
        times = []
        for _ in tqdm(range(N), desc="uploading"):
            start = time()
            # ds.image.append(np.ones((512, 512), dtype="uint8"))
            ds.image.append(np.ones((1024, 1024), dtype="uint8"))
            # ds.image.append(np.ones((28, 28), dtype="uint8"))
            ds.label.append(np.ones((1,), dtype="uint8"))
            times.append((time() - start) * 1000)
            # ds.label.append(y.astype("int32"))
        print("first append times (ms):", times[:10])
        print("last append times (ms):", times[-10:])

        # tensor = ds.image

        # for i in tqdm(range(N), desc="uploading", total=N):
        #     shape = (28, 28, 3)
        #     tensor.append(np.arange(np.prod(shape), dtype=np.uint8).reshape(*shape))


def main():
    cProfile.run("upload()", sort="cumulative")
    # upload()


if __name__ == "__main__":
    main()
