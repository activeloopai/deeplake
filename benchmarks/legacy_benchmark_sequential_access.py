"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import tiledb
import zarr
import hub
import numpy as np
import os
from time import time
from hub.utils import Timer
from tqdm import tqdm


def time_batches(dataset, batch_size=1, hub=False):
    with Timer("Time"):
        counter = 0
        t0 = time()
        for batch in range(dataset.shape[0] // batch_size):
            if hub is True:
                x, y = (
                    dataset[batch * batch_size : (batch + 1) * batch_size][
                        "image"
                    ].compute(),
                    dataset[batch * batch_size : (batch + 1) * batch_size][
                        "label"
                    ].compute(),
                )
            else:
                x, y = (
                    dataset[batch * batch_size : (batch + 1) * batch_size, :-1],
                    dataset[batch * batch_size : (batch + 1) * batch_size, -1],
                )
            counter += 1
            t1 = time()
            print("Batch", counter, f"dt: {t1 - t0}")
            t0 = t1


def time_tiledb(dataset, batch_size=1, split=None):
    if os.path.exists(dataset.split("/")[1] + "_tileDB"):
        ds_tldb = tiledb.open(dataset.split("/")[1] + "_tileDB")
    else:
        if split is not None:
            ds = hub.Dataset(
                dataset + "_" + split, cache=False, storage_cache=False, mode="r"
            )
        else:
            ds = hub.Dataset(dataset, cache=False, storage_cache=False, mode="r")
        y_dim = tiledb.Dim(
            name="y",
            domain=(0, ds.shape[0] - 1),
            tile=500,
            dtype="uint64",
        )
        x_dim = tiledb.Dim(
            name="x",
            domain=(
                0,
                ds["image"].shape[1] * ds["image"].shape[2] * ds["image"].shape[3],
            ),
            tile=ds["image"].shape[1] * ds["image"].shape[2] * ds["image"].shape[3],
            dtype="uint64",
        )
        domain = tiledb.Domain(y_dim, x_dim)
        attr = tiledb.Attr(name="", dtype="int64", var=False)
        schema = tiledb.ArraySchema(
            domain=domain,
            attrs=[attr],
            cell_order="row-major",
            tile_order="row-major",
            sparse=False,
        )
        tiledb.Array.create(dataset.split("/")[1] + "_tileDB", schema)
        ds_tldb = tiledb.open(dataset.split("/")[1] + "_tileDB", mode="w")

        print("Creating TileDB DenseArray:", flush=True)
        for batch in tqdm(range(ds.shape[0] // batch_size)):
            ds_numpy = np.concatenate(
                (
                    ds["image", batch * batch_size : (batch + 1) * batch_size]
                    .compute()
                    .reshape(batch_size, -1),
                    ds["label", batch * batch_size : (batch + 1) * batch_size]
                    .compute()
                    .reshape(batch_size, -1),
                ),
                axis=1,
            )
            ds_tldb[batch * batch_size : (batch + 1) * batch_size] = ds_numpy
        ds_tldb = tiledb.open(dataset.split("/")[1] + "_tileDB", mode="r")

    assert type(ds_tldb) == tiledb.array.DenseArray

    time_batches(ds_tldb, batch_size)


def time_zarr(dataset, batch_size=1, split=None):
    if os.path.exists(dataset.split("/")[1] + "_zarr"):
        ds_zarr = zarr.open(dataset.split("/")[1] + "_zarr")
    else:
        if split is not None:
            ds = hub.Dataset(
                dataset + "_" + split, cache=False, storage_cache=False, mode="r"
            )
        else:
            ds = hub.Dataset(dataset, cache=False, storage_cache=False, mode="r")
        store = zarr.DirectoryStore(dataset.split("/")[1] + "_zarr")
        shape = [
            ds["image"].shape[0],
            ds["image"].shape[1] * ds["image"].shape[2] * ds["image"].shape[3] + 1,
        ]
        ds_zarr = zarr.create(
            (shape[0], shape[1]), store=store, chunks=(batch_size, None)
        )
        print("Creating Zarr Array:", flush=True)
        for batch in tqdm(range(ds.shape[0] // batch_size)):
            ds_numpy = np.concatenate(
                (
                    ds["image", batch * batch_size : (batch + 1) * batch_size]
                    .compute()
                    .reshape(batch_size, -1),
                    ds["label", batch * batch_size : (batch + 1) * batch_size]
                    .compute()
                    .reshape(batch_size, -1),
                ),
                axis=1,
            )
            ds_zarr[batch * batch_size : (batch + 1) * batch_size] = ds_numpy

    assert type(ds_zarr) == zarr.core.Array

    time_batches(ds_zarr, batch_size)


def time_hub(dataset, batch_size=1, split=None):
    if split is not None:
        ds = hub.Dataset(
            dataset + "_" + split, cache=False, storage_cache=False, mode="r"
        )
    else:
        ds = hub.Dataset(dataset, cache=False, storage_cache=False, mode="r")

    assert type(ds) == hub.api.dataset.Dataset

    time_batches(ds, batch_size, hub=True)


configs = [
    {"dataset": "activeloop/mnist", "batch_size": 7000},
    {"dataset": "activeloop/mnist", "batch_size": 70000},
    {"dataset": "hydp/places365_small_train", "batch_size": 1000},
]


if __name__ == "__main__":
    for config in configs:
        dataset = config["dataset"]
        batch_size = config["batch_size"]

        if dataset.split("/")[1].split("_")[-1] == ("train" or "test"):
            dataset = dataset.split("_")
            split = dataset.pop()
            dataset = "_".join(dataset)
        else:
            split = None

        if not os.path.exists(dataset.split("/")[1] + "_hub"):
            if split is not None:
                data = hub.Dataset.from_tfds(dataset.split("/")[1], split=split)
            else:
                data = hub.Dataset.from_tfds(dataset.split("/")[1])
            data.store(dataset.split("/")[1] + "_hub")

        print("Dataset: ", dataset, "with Batch Size: ", batch_size)
        print("Performance of TileDB")
        time_tiledb(dataset, batch_size, split)
        print("Performance of Zarr")
        time_zarr(dataset, batch_size, split)
        print("Performance of Hub (Stored on the Cloud):")
        time_hub(dataset, batch_size, split)
        print("Performance of Hub (Stored Locally):")
        time_hub(dataset.split("/")[1] + "_hub", batch_size, split=None)
