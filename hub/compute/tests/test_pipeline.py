import pytest
import time
import numpy as np

import zarr

import hub

from hub.features import Tensor
from hub.utils import ray_loaded

my_schema = {
    "image": Tensor((28, 28, 4), "int32", (28, 28, 4), chunks=(28, 28, 4)),
    "label": "<U20",
}


@hub.transform(schema=my_schema)
def my_transform(sample):
    return {
        "image": sample["image"].numpy() * 2,
        "label": sample["label"].numpy(),
    }


def test_pipeline_basic():
    ds = hub.Dataset(
        "./data/test/test_pipeline_basic", mode="w", shape=(100,), schema=my_schema, cache=0
    )

    out_ds = my_transform(ds)
    res_ds = out_ds.store("./data/test/test_pipeline_basic_output")
    t2 = time.time()
    
    print("writing", t1 - t0, "transform", t2 - t1)
    exit()
    assert res_ds["label", 5].numpy() == "hello 5"
    assert (res_ds["image", 4].numpy() == 2 * np.ones((28, 28, 4), dtype="int32")).all()


@pytest.mark.skipif(
    not ray_loaded(),
    reason="requires ray to be loaded",
)
def test_pipeline_ray():
    pass

def benchmark():
    arr = np.zeros((100, 28, 28, 4), dtype="int32")
    zarr_arr = zarr.zeros((100, 28, 28, 4), dtype="int32", store=zarr.storage.LMDBStore("./data/array"), overwrite=True)

    t0 = time.time()
    for i in range(100):
        # ds["image", i] = np.ones((28, 28, 4), dtype="int32")
        # ds["label", i] = f"hello {i}"
        # arr[i] = np.ones((28, 28, 4), dtype="int32")
        zarr_arr[i] = np.ones((28, 28, 4), dtype="int32")
    t1 = time.time()
    print(t1-t0)


if __name__ == "__main__":
    # test_pipeline_basic()
    benchmark()