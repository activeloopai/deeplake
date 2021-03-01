"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

from hub.schema.features import SchemaDict
from hub.api.sharded_datasetview import ShardedDatasetView
from hub import Dataset
import pytest


def test_sharded_dataset():
    dt = {"first": "float", "second": "float"}
    datasets = [
        Dataset(schema=dt, shape=(10,), url=f"./data/test/test_dataset/{i}", mode="w")
        for i in range(4)
    ]

    ds = ShardedDatasetView(datasets)

    ds[0]["first"] = 2.3
    assert ds[0]["second"].numpy() != 2.3
    assert ds[30]["first"].numpy() == 0
    assert len(ds) == 40
    assert ds.shape == (40,)
    assert type(ds.schema) == SchemaDict
    assert ds.__repr__() == "ShardedDatasetView(shape=(40,))"
    ds[4, "first"] = 3
    for _ in ds:
        pass

    ds2 = ShardedDatasetView([])
    assert ds2.identify_shard(5) == (0, 0)


def test_sharded_dataset_with_views():
    schema = {"first": "float", "second": "float"}
    ds = Dataset("./data/test_sharded_ds", shape=(10,), schema=schema, mode="w")
    for i in range(10):
        ds[i, "first"] = i
        ds[i, "second"] = 2 * i + 1

    dsv = ds[3:5]
    dsv2 = ds[1]
    dsv3 = ds[8:]
    datasets = [dsv, ds, dsv2, dsv3]
    sharded_ds = ShardedDatasetView(datasets)
    for i in range(2):
        assert sharded_ds[i, "first"].compute() == i + 3
        assert sharded_ds[i, "second"].compute() == 2 * (i + 3) + 1
    for i in range(2, 12):
        assert sharded_ds[i, "first"].compute() == i - 2
        assert sharded_ds[i, "second"].compute() == 2 * (i - 2) + 1
    assert sharded_ds[12, "first"].compute() == 1
    assert sharded_ds[12, "second"].compute() == 3
    for i in range(13, 15):
        assert sharded_ds[i, "first"].compute() == i - 5
        assert sharded_ds[i, "second"].compute() == 2 * (i - 5) + 1


def test_sharded_dataset_advanced_slice():
    schema = {"first": "float", "second": "float"}
    ds = Dataset("./data/test_sharded_ds", shape=(10,), schema=schema, mode="w")
    for i in range(10):
        ds[i, "first"] = i
        ds[i, "second"] = 2 * i + 1

    dsv = ds[3:5]
    dsv2 = ds[1]
    dsv3 = ds[8:]
    datasets = [dsv, ds, dsv2, dsv3]
    sharded_ds = ShardedDatasetView(datasets)
    assert sharded_ds["first", :].compute().tolist() == [
        3,
        4,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        1,
        8,
        9,
    ]
    assert sharded_ds["first"].compute().tolist() == [
        3,
        4,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        1,
        8,
        9,
    ]
    assert sharded_ds["first", -4:].compute().tolist() == [9, 1, 8, 9]
    assert sharded_ds[1:3].compute()[0] == {"first": 4.0, "second": 9.0}
    assert sharded_ds[1:3].compute()[1] == {"first": 0.0, "second": 1.0}
    sharded_ds["first", 1:5] = [10, 11, 12, 13]
    assert sharded_ds["first", 1:5].compute().tolist() == [10, 11, 12, 13]
    sharded_ds["first", 12] = 50
    assert sharded_ds["first", 12].compute() == 50


if __name__ == "__main__":
    # test_sharded_dataset()
    test_sharded_dataset_advanced_slice()
