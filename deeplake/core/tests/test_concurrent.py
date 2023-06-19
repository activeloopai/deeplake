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
        f"Hello from worker {worker_id}! len(ds) = {len(ds)}, len(ds._images_id) = {len(ds._images_id)}"
    )
    assert len(ds.images) == len(ds._images_id), (len(ds.images), len(ds._images_id))
    if checkout:
        ds.checkout("abcd")
        assert ds.images.meta.links
    assert len(ds.images) == len(ds._images_id), (len(ds.images), len(ds._images_id))
    with ds.concurrent():
        assert ds.images.meta.links
        assert len(ds.images) == len(ds._images_id), (
            len(ds.images),
            len(ds._images_id),
        )
        n = len(ds.images)
        for i in tqdm.tqdm(range(10)):
            ds.images.extend(rand_images())
            ds.labels.extend(rand_labels())
        assert ds.images._extend_links
        assert ds.images.meta.links
        assert len(ds.images) == len(ds._images_id), (
            n,
            len(ds.images),
            len(ds._images_id),
        )
    ds.checkout("main")
    assert len(ds.images) == len(ds._images_id), (len(ds.images), len(ds._images_id))
    print(f"Worker {worker_id} finished! len(ds) = {len(ds)}")
    return len(ds)


@pytest.mark.parametrize("commit", [True])
@pytest.mark.parametrize("mode", ["sync", "async", "overlap"])
def test_concurrent(commit, mode):
    ds = deeplake.empty(ds_path, overwrite=True)
    assert not ds.commit_id
    assert ds.is_head_node
    with ds:
        ds.create_tensor("images", htype="image", sample_compression="jpeg")
        ds.create_tensor("labels", htype="class_label")
        if commit:
            ds.commit(hash="abcd")
        assert len(ds.images) == len(ds._images_id), (
            len(ds.images),
            len(ds._images_id),
        )
    executor = ProcessPoolExecutor()
    nsamples = list(executor.map(worker, range(5), repeat(commit), repeat(mode)))
    if mode == "async":
        assert sorted(nsamples) == nsamples
    assert min(nsamples) == 100
    ds = deeplake.load(ds_path)
    assert len(ds) == 500
    assert ds.branches == ["main"]
    deeplake.delete(ds_path)
