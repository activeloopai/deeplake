import deeplake


import numpy as np
import tqdm
import time
import pytest
from itertools import repeat

from concurrent.futures import ProcessPoolExecutor

ds_path = "concurrent_test_ds/"


def rand_images():
    return np.random.randint(0, 255, size=(10, 256, 256, 3), dtype=np.uint8)


def rand_labels():
    return np.random.randint(0, 10, size=(10,), dtype=np.uint8)


def worker(worker_id: int, checkout: bool, mode: bool):
    if mode == "async":
        time.sleep(worker_id)
    elif mode == "overlap":
        time.sleep(worker_id * 0.1)
    ds = deeplake.load(ds_path)
    assert ds.images.meta.links
    print(
        f"Hello from worker {worker_id}!"
    )
    if checkout:
        ds.checkout("abcd")
    with ds.concurrent():
        for i in tqdm.tqdm(range(10)):
            ds.images.extend(rand_images())
            ds.labels.extend(rand_labels())
    ds.checkout("main")
    print(f"Worker {worker_id} done.")
    return len(ds)


@pytest.mark.parametrize("commit", [True])
@pytest.mark.parametrize("mode", ["sync", "async", "overlap"])
def test_concurrent(commit, mode):
    ds = deeplake.empty(ds_path, overwrite=True)
    with ds:
        ds.create_tensor("images", htype="image", sample_compression="jpeg")
        ds.create_tensor("labels", htype="class_label")
        if commit:
            ds.commit(hash="abcd")
    executor = ProcessPoolExecutor()
    nsamples = list(executor.map(worker, range(5), repeat(commit), repeat(mode)))
    if mode == "async":
        assert sorted(nsamples) == nsamples
    assert min(nsamples) == 100
    ds = deeplake.load(ds_path)
    assert len(ds) == 500
    assert ds.branches == ["main"]
    for i, sample in enumerate(ds):
        sample.images.numpy()
        sample.labels.numpy()
    deeplake.delete(ds_path)
