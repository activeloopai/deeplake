import numpy as np
import hub
import pytest
from hub.util.json import JsonValidationError


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
        assert ds.json[i].numpy()[0] == items[i % 2]


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
        assert ds.json[i].numpy()[0]["y"] == items[i % 2]["y"]
        np.testing.assert_array_equal(ds.json[i].numpy()[0]["x"], items[i % 2]["x"])


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
        assert ds.json[i].numpy()[0] == items[i % 2]


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
        assert list(ds.list[i].numpy()) == items[i % 2]
