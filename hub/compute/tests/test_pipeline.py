import numpy as np
import pytest
import zarr

import hub
from hub.features import Tensor
from hub.utils import ray_loaded, pathos_loaded, Timer

try:
    import ray
except:
    pass
my_schema = {
    "image": Tensor((28, 28, 4), "int32", (28, 28, 4)),
    "label": "<U20",
}


@hub.transform(schema=my_schema)
def my_transform(sample, multiplier: int = 2):
    return {
        "image": sample["image"].numpy() * multiplier,
        "label": sample["label"].numpy(),
    }


def test_pipeline_basic():
    ds = hub.Dataset(
        "./data/test/test_pipeline_basic", mode="w", shape=(100,), schema=my_schema
    )
    
    for i in range(len(ds)):
        ds["image", i] = np.ones((28, 28, 4), dtype="int32")
        ds["label", i] = f"hello {i}"

    out_ds = my_transform(ds, multiplier=2)
    res_ds = out_ds.store("./data/test/test_pipeline_basic_output")
    
    assert res_ds["label", 5].numpy() == "hello 5"
    assert (res_ds["image", 4].numpy() == 2 * np.ones((28, 28, 4), dtype="int32")).all()
    assert len(res_ds) == len(out_ds)
    assert res_ds.shape[0] == out_ds.shape[0] 
    assert "image" in res_ds.schema.dict_ and "label" in res_ds.schema.dict_

@pytest.mark.skipif(
    not ray_loaded(),
    reason="requires ray to be loaded",
)
def test_pipeline_ray():
    pass


@pytest.mark.skipif(
    not pathos_loaded(),
    reason="requires pathos to be loaded",
)
def test_pathos(sample_size=100, width=100, channels=4, dtype="uint8"):

    my_schema = {
        "image": Tensor((width, width, channels), dtype, (width, width, channels), chunks=(sample_size // 20, width, width, channels)),
    }

    with Timer("pathos"):
        @hub.transform(schema=my_schema, scheduler="pathos", processes=1)
        def my_transform(x):
            return {
                "image": (np.ones((width, width, channels), dtype=dtype) * 255),
            }
        
        ds = hub.Dataset(
            "./data/test/test_pipeline_basic_3", mode="w", shape=(sample_size,), schema=my_schema, cache=0
        )

        ds_t = my_transform(ds).store("./data/test/test_pipeline_basic_4")

    assert (ds_t["image", :].numpy() == 255).all()

def benchmark(sample_size=100, width=1000, channels=4, dtype="int8"):
    numpy_arr = np.zeros((sample_size, width, width, channels), dtype=dtype)
    zarr_fs = zarr.zeros((sample_size, width, width, channels), dtype=dtype, store=zarr.storage.FSStore("./data/test/array"), overwrite=True)
    zarr_lmdb = zarr.zeros((sample_size, width, width, channels), dtype=dtype, store=zarr.storage.LMDBStore("./data/test/array2"), overwrite=True)
    
    my_schema = {
        "image": Tensor((width, width, channels), dtype, (width, width, channels)),
    }

    ds_fs = hub.Dataset(
        "./data/test/test_pipeline_basic_3", mode="w", shape=(sample_size,), schema=my_schema, cache=0
    )

    ds_fs_cache = hub.Dataset(
        "./data/test/test_pipeline_basic_2", mode="w", shape=(sample_size,), schema=my_schema
    )
    if False:
        print(f"~~~ Sequential write of {sample_size}x{width}x{width}x{channels} random arrays ~~~")
        for name, arr in [("Numpy", numpy_arr), ("Zarr FS", zarr_fs), 
                        ("Zarr LMDB", zarr_lmdb), ("Hub FS", ds_fs["image"]), 
                        ("Hub FS+Cache", ds_fs_cache["image"])]:
            with Timer(name):
                for i in range(sample_size):
                    arr[i] = (np.random.rand(width, width, channels) * 255).astype(dtype)

    print(f"~~~ Pipeline {sample_size}x{width}x{width}x{channels} random arrays ~~~")
    for name, processes in [("single", 1), ("pathos", 10)]:  # , ("ray", 10), ("green", 10), ("dask", 10)]:
        @hub.transform(schema=my_schema, scheduler=name, processes=processes)
        def my_transform(sample):
            return {
                "image": (np.random.rand(width, width, channels) * 255).astype(dtype),
            }

        with Timer(name):
            out_ds = my_transform(ds_fs)
            res_ds = out_ds.store(f"./data/test/test_pipeline_basic_output_{name}")

if __name__ == "__main__":
    # test_pipeline_basic()
    test_pathos()
    # benchmark()