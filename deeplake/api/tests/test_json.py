import numpy as np
import hub
import pytest
from hub.util.json import JsonValidationError
from hub.tests.dataset_fixtures import (
    enabled_non_gcs_gdrive_datasets,
    enabled_non_gcs_datasets,
)
from typing import Any, Optional, Union, List, Dict


def test_json_basic(memory_ds):
    ds = memory_ds
    ds.create_tensor("json", htype="json")
    items = [
        {"x": [1, 2, 3], "y": [4, [5, 6]]},
        {"x": [1, 2, 3], "y": [4, {"z": [0.1, 0.2, []]}]},
    ]
    with ds:
        for x in items:
            ds.json.append(x)
        ds.json.extend(items)
    assert ds.json.shape == (4, 1)
    for i in range(4):
        assert ds.json[i].data()["value"] == items[i % 2]
        assert ds.json[i].dict() == items[i % 2]
    assert ds.json.dict() == items * 2


def test_json_with_numpy(memory_ds):
    ds = memory_ds
    ds.create_tensor("json", htype="json")
    items = [
        {"x": np.array([1, 2, 3], dtype=np.float32), "y": [4, [5, 6]]},
        {"x": np.array([1, 2, 3], dtype=np.uint8), "y": [4, {"z": [0.1, 0.2, []]}]},
    ]
    with ds:
        for x in items:
            ds.json.append(x)
        ds.json.extend(items)
    for i in range(4):
        assert ds.json[i].data()["value"]["y"] == items[i % 2]["y"]
        np.testing.assert_array_equal(
            ds.json[i].data()["value"]["x"], items[i % 2]["x"]
        )


def test_json_with_hub_sample(memory_ds, compressed_image_paths):
    ds = memory_ds
    ds.create_tensor("json", htype="json")
    items = [
        {
            "x": [1, 2, 3],
            "y": [4, [5, 6]],
            "z": hub.read(compressed_image_paths["jpeg"][0]),
        },
        {
            "x": [1, 2, 3],
            "y": [4, {"z": [0.1, 0.2, []]}],
            "z": hub.read(compressed_image_paths["png"][0]),
        },
    ]
    with ds:
        for x in items:
            ds.json.append(x)
        ds.json.extend(items)
    assert ds.json.shape == (4, 1)
    for i in range(4):
        assert ds.json[i].data()["value"] == items[i % 2]


def test_json_list_basic(memory_ds):
    ds = memory_ds
    ds.create_tensor("list", htype="list")
    items = [
        [{"x": [1, 2, 3], "y": [4, [5, 6]]}, [[]], [None, 0.1]],
        [[], [[[]]], {"a": [0.1, 1, "a", []]}],
    ]
    with ds:
        for x in items:
            ds.list.append(x)
        ds.list.extend(items)
    assert ds.list.shape == (4, 3)
    for i in range(4):
        assert ds.list[i].data()["value"] == items[i % 2]
    for i, x in enumerate(ds.list.data()["value"]):
        assert x == items[i % 2]


def test_list_with_numpy(memory_ds):
    ds = memory_ds
    ds.create_tensor("list", htype="list")
    items = [
        [
            np.random.random((3, 4)),
            {"x": [1, 2, 3], "y": [4, [5, 6]]},
            [[]],
            [None, 0.1],
        ],
        [np.random.randint(0, 10, (4, 5)), [], [[[]]], {"a": [0.1, 1, "a", []]}],
    ]
    with ds:
        for x in items:
            ds.list.append(x)
        ds.list.extend(items)
    assert ds.list.shape == (4, 4)
    for i in range(4):
        actual, expected = ds.list[i].data()["value"], items[i % 2]
        np.testing.assert_array_equal(actual[0], expected[0])
        assert actual[1:] == expected[1:]


def test_list_with_hub_sample(memory_ds, compressed_image_paths):
    ds = memory_ds
    ds.create_tensor("list", htype="list")
    items = [
        [
            {
                "x": [1, 2, 3],
                "y": [4, [5, 6, hub.read(compressed_image_paths["jpeg"][0])]],
            },
            [[hub.read(compressed_image_paths["jpeg"][1])]],
            [None, 0.1],
        ],
        [
            [],
            [[[hub.read(compressed_image_paths["png"][0])]]],
            {"a": [0.1, 1, "a", hub.read(compressed_image_paths["png"][0])]},
        ],
    ]
    with ds:
        for x in items:
            ds.list.append(x)
        ds.list.extend(items)
    assert ds.list.shape == (4, 3)
    for i in range(4):
        assert ds.list[i].data()["value"] == items[i % 2]


def test_json_with_schema(memory_ds):
    ds = memory_ds
    ds.create_tensor("json", htype="json", dtype=List[Dict[str, int]])
    ds.json.append([{"x": 1, "y": 2}])
    with pytest.raises(JsonValidationError):
        ds.json.append({"x": 1, "y": 2})
    with pytest.raises(JsonValidationError):
        ds.json.append([{"x": 1, "y": "2"}])

    assert ds.json.numpy()[0, 0] == [{"x": 1, "y": 2}]

    ds.create_tensor("json2", htype="json", dtype=Optional[List[Dict[str, int]]])
    items = [
        [{"x": 1, "y": 2}],
        None,
        [{"x": 2, "y": 3}],
        None,
    ]
    ds.json2.extend(items)
    for i in range(len(items)):
        assert (
            ds.json2[i].data()["value"]
            == ds.json2.data()["value"][i]
            == (items[i] or {})
        )


@enabled_non_gcs_datasets
@pytest.mark.parametrize("compression", ["lz4", None])
def test_json_transform(ds, compression, scheduler="threaded"):
    ds.create_tensor("json", htype="json", sample_compression=compression)

    items = [
        {"x": [1, 2, 3], "y": [4, [5, 6]]},
        {"x": [1, 2, 3], "y": [4, {"z": [0.1, 0.2, []]}]},
        ["a", ["b", "c"], {"d": 1.0}],
        [1.0, 2.0, 3.0, 4.0],
        ["a", "b", "c", "d"],
        1,
        5.0,
        True,
        False,
        None,
    ] * 5

    @hub.compute
    def upload(stuff, ds):
        ds.json.append(stuff)
        return ds

    upload().eval(items, ds, num_workers=2, scheduler=scheduler)
    assert ds.json.data()["value"] == items
    with pytest.raises(Exception):
        ds.json.list()


@enabled_non_gcs_gdrive_datasets
def test_list_transform(ds, scheduler="threaded"):
    ds.create_tensor("list", htype="list")

    items = [
        ["a", ["b", "c"], {"d": 1.0}],
        [1.0, 2.0, 3.0, 4.0],
        ["a", "b", "c", "d"],
    ] * 5

    @hub.compute
    def upload(stuff, ds):
        ds.list.append(stuff)
        return ds

    upload().eval(items, ds, num_workers=2, scheduler=scheduler)
    assert ds.list.data()["value"] == items
    assert ds.list[0].list() == items[0]
    assert ds.list.list() == items
