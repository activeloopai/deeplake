import deeplake


import numpy as np
import tqdm

from concurrent.futures import ProcessPoolExecutor

ds_path = "concurrent_test_ds/"


def rand_images():
    return np.random.randint(0, 255, size=(10, 256, 256, 3), dtype=np.uint8)


def rand_labels():
    return np.random.randint(0, 10, size=(10,), dtype=np.uint8)


def worker(worker_id: int):
    import time

    # time.sleep(worker_id / 10)
    ds = deeplake.load(ds_path)
    print(f"Hello from worker {worker_id}!")
    with ds.concurrent():
        for i in tqdm.tqdm(range(10)):
            ds.images.extend(rand_images())
            ds.labels.extend(rand_labels())
        return len(ds)


def test_concurrent():
    ds = deeplake.empty(ds_path, overwrite=True)
    with ds:
        ds.create_tensor("images", htype="image", sample_compression="jpeg")
        ds.create_tensor("labels", htype="class_label")
        ds.commit()
    executor = ProcessPoolExecutor()
    nsamples = list(executor.map(worker, range(5)))
    assert nsamples == [100] * 5
    ds = deeplake.load(ds_path)
    assert len(ds) == 500
    assert ds.branches == ["main"]
    deeplake.delete(ds_path)
