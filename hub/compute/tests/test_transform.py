"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import numpy as np
import zarr
import os
from hub.cli.auth import login_fn
import hub
from hub.schema import Tensor, Image, Text
from hub.utils import Timer
from hub.utils import hub_creds_exist
import pytest

my_schema = {
    "image": Tensor((28, 28, 4), "int32", (28, 28, 4)),
    "label": Text((None,), "int64", (20,)),
    "confidence": "float",
}

dynamic_schema = {
    "image": Tensor(shape=(None, None, None), dtype="int32", max_shape=(32, 32, 3)),
    "label": Text((None,), "int64", (20,)),
}


def test_pipeline_basic():
    ds = hub.Dataset(
        "./data/test/test_pipeline_basic", mode="w", shape=(100,), schema=my_schema
    )

    for i in range(len(ds)):
        ds["image", i] = np.ones((28, 28, 4), dtype="int32")
        ds["label", i] = f"hello {i}"
        ds["confidence", i] = 0.2

    @hub.transform(schema=my_schema)
    def my_transform(sample, multiplier: int = 2):
        return {
            "image": sample["image"] * multiplier,
            "label": sample["label"],
            "confidence": sample["confidence"] * multiplier,
        }

    out_ds = my_transform(ds, multiplier=2)
    assert (out_ds["image", 0].compute() == 2).all()
    assert len(list(out_ds)) == 100
    res_ds = out_ds.store("./data/test/test_pipeline_basic_output")
    assert res_ds["label", 5].compute() == "hello 5"
    assert (
        res_ds["image", 4].compute() == 2 * np.ones((28, 28, 4), dtype="int32")
    ).all()
    assert len(res_ds) == len(out_ds)
    assert res_ds.shape[0] == out_ds.shape[0]
    assert "image" in res_ds.schema.dict_ and "label" in res_ds.schema.dict_


def test_threaded():
    init_schema = {
        "image": Tensor(
            shape=(None, None, None), max_shape=(4, 224, 224), dtype="float32"
        )
    }
    schema = {
        "image": Tensor(
            shape=(None, None, None), max_shape=(4, 224, 224), dtype="float32"
        ),
        "label": Tensor(shape=(None,), max_shape=(6,), dtype="uint8"),
        "text_label": Text((None,), "int64", (14,)),
        "flight_code": Text((None,), "int64", (10,)),
    }

    ds_init = hub.Dataset(
        "./data/hub/new_pipeline_threaded2",
        mode="w",
        shape=(10,),
        schema=init_schema,
        cache=False,
    )

    for i in range(len(ds_init)):
        ds_init["image", i] = np.ones((4, 220, 224))
        ds_init["image", i] = np.ones((4, 221, 224))

    @hub.transform(schema=schema, scheduler="threaded", workers=2)
    def create_classification_dataset(sample):
        ts = sample["image"]
        return [
            {
                "image": ts,
                "label": np.ones((6,)),
                "text_label": "PLANTED",
                "flight_code": "UYKNTHNXR",
            }
            for _ in range(5)
        ]

    ds = create_classification_dataset(ds_init).store(
        "./data/hub/new_pipeline_threaded_final"
    )

    assert ds["image", 0].shape[1] == 221


def test_pipeline_dynamic():
    ds = hub.Dataset(
        "./data/test/test_pipeline_dynamic3",
        mode="w",
        shape=(1,),
        schema=dynamic_schema,
        cache=False,
    )

    ds["image", 0] = np.ones((30, 32, 3))

    @hub.transform(schema=dynamic_schema)
    def dynamic_transform(sample, multiplier: int = 2):
        return {
            "image": sample["image"] * multiplier,
            "label": sample["label"],
        }

    out_ds = dynamic_transform(ds, multiplier=4).store(
        "./data/test/test_pipeline_dynamic_output2"
    )

    assert (
        out_ds["image", 0].compute() == 4 * np.ones((30, 32, 3), dtype="int32")
    ).all()


def test_pipeline_multiple():
    ds = hub.Dataset(
        "./data/test/test_pipeline_dynamic3",
        mode="w",
        shape=(1,),
        schema=dynamic_schema,
        cache=False,
    )

    ds["image", 0] = np.ones((30, 32, 3))

    @hub.transform(schema=dynamic_schema, scheduler="threaded", workers=2)
    def dynamic_transform(sample, multiplier: int = 2):
        return [
            {
                "image": sample["image"] * multiplier,
                "label": sample["label"],
            }
            for i in range(4)
        ]

    out_ds = dynamic_transform(ds, multiplier=4).store(
        "./data/test/test_pipeline_dynamic_output2"
    )
    assert len(out_ds) == 4
    assert (
        out_ds["image", 0].compute() == 4 * np.ones((30, 32, 3), dtype="int32")
    ).all()


def test_multiprocessing(sample_size=200, width=100, channels=4, dtype="uint8"):

    my_schema = {
        "image": Image(
            (width, width, channels),
            dtype,
            (width, width, channels),
            chunks=(sample_size // 20),
            compressor="LZ4",
        ),
    }

    with Timer("multiprocesing"):

        @hub.transform(schema=my_schema, scheduler="threaded", workers=4)
        def my_transform(x):

            a = np.random.random((width, width, channels))
            for i in range(100):
                a *= np.random.random((width, width, channels))

            return {
                "image": (np.ones((width, width, channels), dtype=dtype) * 255),
            }

        ds = hub.Dataset(
            "./data/test/test_pipeline_basic_4",
            mode="w",
            shape=(sample_size,),
            schema=my_schema,
            cache=2 * 26,
        )

        ds_t = my_transform(ds).store("./data/test/test_pipeline_basic_4")

    assert (ds_t["image", :].compute() == 255).all()


def test_pipeline():

    ds = hub.Dataset(
        "./data/test/test_pipeline_multiple2", mode="w", shape=(100,), schema=my_schema
    )

    for i in range(len(ds)):
        ds["image", i] = np.ones((28, 28, 4), dtype="int32")
        ds["label", i] = f"hello {i}"
        ds["confidence", i] = 0.2

    with Timer("multiple pipes"):

        @hub.transform(schema=my_schema)
        def my_transform(sample, multiplier: int = 2):
            return {
                "image": sample["image"] * multiplier,
                "label": sample["label"],
                "confidence": sample["confidence"] * multiplier,
            }

        out_ds = my_transform(ds, multiplier=2)
        out_ds = my_transform(out_ds, multiplier=2)
        out_ds = out_ds.store("./data/test/test_pipeline_multiple_4")

        assert (out_ds["image", 0].compute() == 4).all()


def test_stacked_transform():
    schema = {"test": Tensor((2, 2), dtype="uint8")}

    @hub.transform(schema=schema)
    def multiply_transform(sample, multiplier=1, times=1):
        if times == 1:
            return {"test": multiplier * sample["test"]}
        else:
            return [{"test": multiplier * sample["test"]} for i in range(times)]

    @hub.transform(schema=schema)
    def multiply_transform_2(sample, multiplier=1, times=1):
        if times == 1:
            return {"test": multiplier * sample["test"]}
        else:
            return [{"test": multiplier * sample["test"]} for i in range(times)]

    ds = hub.Dataset("./data/stacked_transform", mode="w", shape=(5,), schema=schema)
    for i in range(5):
        ds["test", i] = np.ones((2, 2))
    ds1 = multiply_transform(ds, multiplier=2, times=5)
    ds2 = multiply_transform(ds1, multiplier=3, times=2)
    ds3 = multiply_transform_2(ds2, multiplier=5, times=3)
    ds4 = ds3.store("./data/stacked_transform_2")
    assert len(ds4) == 150
    assert (ds4["test", 0].compute() == 30 * np.ones((2, 2))).all()


def test_text():
    my_schema = {"text": Text((None,), max_shape=(10,))}

    @hub.transform(schema=my_schema)
    def my_transform(sample):
        return {"text": np.array("abc")}

    ds = my_transform([i for i in range(10)])
    ds2 = ds.store("./data/test/transform_text")
    for i in range(10):
        assert ds2["text", i].compute() == "abc"


def test_zero_sample_transform():
    schema = {"test": Tensor((None, None), dtype="uint8", max_shape=(10, 10))}

    @hub.transform(schema=schema)
    def my_transform(sample):
        if sample % 5 == 0:
            return []
        else:
            return {"test": (sample % 5) * np.ones((5, 5))}

    ds = my_transform([i for i in range(30)])
    ds2 = ds.store("./data/transform_zero_sample", sample_per_shard=5)
    assert len(ds2) == 24
    for i, item in enumerate(ds2):
        assert (item["test"].compute() == (((i % 4) + 1) * np.ones((5, 5)))).all()


def test_mutli_sample_transform():
    schema = {"test": Tensor((None, None), dtype="uint8", max_shape=(10, 10))}

    @hub.transform(schema=schema)
    def my_transform(sample):
        d = {"test": sample * np.ones((5, 5))}
        return [d, d]

    ds = my_transform([i for i in range(32)])
    ds2 = ds.store("./data/transform_zero_sample", sample_per_shard=5)
    assert len(ds2) == 64
    for i, item in enumerate(ds2):
        assert (item["test"].compute() == (i // 2) * np.ones((5, 5))).all()


def benchmark(sample_size=100, width=1000, channels=4, dtype="int8"):
    numpy_arr = np.zeros((sample_size, width, width, channels), dtype=dtype)
    zarr_fs = zarr.zeros(
        (sample_size, width, width, channels),
        dtype=dtype,
        store=zarr.storage.FSStore("./data/test/array"),
        overwrite=True,
    )
    zarr_lmdb = zarr.zeros(
        (sample_size, width, width, channels),
        dtype=dtype,
        store=zarr.storage.LMDBStore("./data/test/array2"),
        overwrite=True,
    )

    my_schema = {
        "image": Tensor((width, width, channels), dtype, (width, width, channels)),
    }

    ds_fs = hub.Dataset(
        "./data/test/test_pipeline_basic_3",
        mode="w",
        shape=(sample_size,),
        schema=my_schema,
        cache=0,
    )

    ds_fs_cache = hub.Dataset(
        "./data/test/test_pipeline_basic_2",
        mode="w",
        shape=(sample_size,),
        schema=my_schema,
    )
    if False:
        print(
            f"~~~ Sequential write of {sample_size}x{width}x{width}x{channels} random arrays ~~~"
        )
        for name, arr in [
            ("Numpy", numpy_arr),
            ("Zarr FS", zarr_fs),
            ("Zarr LMDB", zarr_lmdb),
            ("Hub FS", ds_fs["image"]),
            ("Hub FS+Cache", ds_fs_cache["image"]),
        ]:
            with Timer(name):
                for i in range(sample_size):
                    arr[i] = (np.random.rand(width, width, channels) * 255).astype(
                        dtype
                    )

    print(f"~~~ Pipeline {sample_size}x{width}x{width}x{channels} random arrays ~~~")
    for name, processes in [
        ("single", 1),
        ("processed", 10),
    ]:  # , ("ray", 10), ("green", 10), ("dask", 10)]:

        @hub.transform(schema=my_schema, scheduler=name, processes=processes)
        def my_transform(sample):
            return {
                "image": (np.random.rand(width, width, channels) * 255).astype(dtype),
            }

        with Timer(name):
            out_ds = my_transform(ds_fs)
            out_ds.store(f"./data/test/test_pipeline_basic_output_{name}")


@pytest.mark.skipif(not hub_creds_exist(), reason="requires hub credentials")
def test_transform_overwrite():
    password = os.getenv("ACTIVELOOP_HUB_PASSWORD")
    login_fn("testingacc", password)

    schema = {
        "image": hub.schema.Image(
            shape=(None, None, 1), dtype="uint8", max_shape=(1, 1, 1)
        ),
    }

    @hub.transform(schema=schema)
    def create_image(value):
        return {"image": np.ones((1, 1, 1), dtype="uint8") * value}

    ds1 = create_image(range(5))
    ds = ds1.store("testingacc/ds_transform", public=False)
    for i in range(5):
        assert (ds["image", i].compute() == i * np.ones((1, 1, 1))).all()
    ds = ds1.store("testingacc/ds_transform", public=False)
    for i in range(5):
        assert (ds["image", i].compute() == i * np.ones((1, 1, 1))).all()


if __name__ == "__main__":
    with Timer("Test Transform"):
        with Timer("test threaded"):
            test_threaded()

        with Timer("test pipeline"):
            test_pipeline()

        with Timer("test multiprocessing"):
            test_multiprocessing()

        with Timer("test Pipeline"):
            test_pipeline_basic()

        with Timer("test Pipeline Dynamic"):
            test_pipeline_dynamic()
