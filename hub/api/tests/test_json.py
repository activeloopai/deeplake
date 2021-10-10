import numpy as np
import hub
import pytest


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
