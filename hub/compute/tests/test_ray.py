"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import hub
from hub.utils import ray_loaded
from hub.schema import Tensor, Text
import pytest
from hub.compute.ray import empty_remote

import numpy as np

dynamic_schema = {
    "image": Tensor(shape=(None, None, None), dtype="int32", max_shape=(32, 32, 3)),
    "label": Text((None,), "int64", (20,)),
    "confidence": {"confidence": "float"},
}

my_schema = {
    "image": Tensor((28, 28, 4), "int32", (28, 28, 4)),
    "label": Text((None,), "int64", (20,)),
    "confidence": {"confidence": "float"},
}


def test_wrapper():
    @empty_remote
    def a(x):
        return x

    assert a(5)() == 5


@pytest.mark.skipif(
    not ray_loaded(),
    reason="requires ray to be loaded",
)
def test_ray_simple():
    schema = {"var": "float"}

    @hub.transform(schema=schema, scheduler="ray")
    def process(item):
        return {"var": 1}

    ds = process([1, 2, 3]).store("./data/somedataset")
    assert ds["var", 0].compute() == 1


@pytest.mark.skipif(
    not ray_loaded(),
    reason="requires ray to be loaded",
)
def test_ray_non_dynamic():
    schema = {
        "var": Tensor(shape=(2, 2), dtype="uint8"),
        "var2": Tensor(shape=(2, 2), dtype="uint8"),
    }

    @hub.transform(schema=schema, scheduler="ray_generator")
    def process(item):
        return [{"var": np.ones((2, 2)), "var2": np.ones((2, 2))} for i in range(2)]

    ds = process([1, 2, 3]).store("./data/somedataset")
    assert ds["var", 0].compute().shape[0] == 2


@pytest.mark.skipif(
    not ray_loaded(),
    reason="requires ray to be loaded",
)
def test_ray_dynamic():
    schema = {"var": Tensor(shape=(None, None), max_shape=(2, 2), dtype="uint8")}

    @hub.transform(schema=schema, scheduler="ray_generator")
    def process(item):
        return {"var": np.ones((1, 2))}

    ds = process([1, 2, 3]).store("./data/somedataset")
    assert ds["var", 0].compute().shape[0] == 1


@pytest.mark.skipif(
    not ray_loaded(),
    reason="requires ray to be loaded",
)
def test_ray_simple_generator():
    schema = {"var": "float"}

    @hub.transform(schema=schema, scheduler="ray_generator")
    def process(item):
        items = [{"var": item} for i in range(item)]
        return items

    ds = process([0, 1, 2, 3]).store("./data/somegeneratordataset")
    assert ds["var", 0].compute() == 1
    assert ds.shape[0] == 6


@pytest.mark.skipif(
    not ray_loaded(),
    reason="requires ray to be loaded",
)
def test_pipeline_ray():
    ds = hub.Dataset(
        "./data/test/test_pipeline_basic",
        mode="w",
        shape=(100,),
        schema=my_schema,
        cache=False,
    )

    for i in range(len(ds)):
        ds["image", i] = np.ones((28, 28, 4), dtype="int32")
        ds["label", i] = f"hello {i}"
        ds["confidence/confidence", i] = 0.2

    @hub.transform(schema=my_schema, scheduler="ray")
    def my_transform(sample, multiplier: int = 2):
        return {
            "image": sample["image"] * multiplier,
            "label": sample["label"],
            "confidence": {
                "confidence": sample["confidence"]["confidence"] * multiplier
            },
        }

    out_ds = my_transform(ds, multiplier=2)
    assert (out_ds["image", 0].compute() == 2).all()
    assert len(list(out_ds)) == 100
    out_ds.store("./data/test/test_pipeline_basic_output")


@pytest.mark.skipif(
    not ray_loaded(),
    reason="requires ray to be loaded",
)
def test_ray_pipeline_multiple():

    ds = hub.Dataset(
        "./data/test/test_pipeline_dynamic10",
        mode="w",
        shape=(10,),
        schema=dynamic_schema,
        cache=False,
    )

    ds["image", 0] = np.ones((30, 32, 3))

    @hub.transform(schema=dynamic_schema, scheduler="ray_generator", workers=2)
    def dynamic_transform(sample, multiplier: int = 2):
        return [
            {
                "image": sample["image"] * multiplier,
                "label": sample["label"],
            }
            for _ in range(4)
        ]

    out_ds = dynamic_transform(ds, multiplier=4).store(
        "./data/test/test_pipeline_dynamic_output2"
    )
    assert len(out_ds) == 40
    assert (
        out_ds["image", 0].compute() == 4 * np.ones((30, 32, 3), dtype="int32")
    ).all()


@pytest.mark.skipif(
    not ray_loaded(),
    reason="requires ray to be loaded",
)
def test_stacked_transform():
    schema = {"test": Tensor((2, 2), dtype="uint8")}

    @hub.transform(schema=schema)
    def multiply_transform(sample, multiplier=1, times=1):
        if times == 1:
            return {"test": multiplier * sample["test"]}
        else:
            return [{"test": multiplier * sample["test"]} for i in range(times)]

    @hub.transform(schema=schema, scheduler="ray_generator")
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


if __name__ == "__main__":
    test_ray_simple()
    test_ray_non_dynamic()
    test_ray_dynamic()
    test_ray_simple_generator()
    test_pipeline_ray()
    test_ray_pipeline_multiple()
