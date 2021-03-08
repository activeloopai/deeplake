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


def time_batches(dataset, batch_size=1, num_batches=1, hub=False):
    np.random.seed(0)
    with Timer("Time"):
        counter = 0
        t0 = time()
        for batch in range(num_batches):
            if hub is False:
                dataset[
                    batch * batch_size : (batch + 1) * batch_size, :-1
                ] = np.random.randint(255, size=(batch_size, 28 * 28))
                dataset[
                    batch * batch_size : (batch + 1) * batch_size, -1
                ] = np.random.randint(10, size=(batch_size,))
            else:
                dataset["image"][
                    batch * batch_size : (batch + 1) * batch_size
                ] = np.random.randint(255, size=(batch_size, 28, 28, 1))
                dataset["label"][
                    batch * batch_size : (batch + 1) * batch_size
                ] = np.random.randint(10, size=(batch_size, 1))
                dataset.flush()
            counter += 1
            t1 = time()
            print("Batch", counter, f"dt: {t1 - t0}")
            t0 = t1


def time_tiledb(dataset, batch_size=1, num_batches=1):
    if os.path.exists(dataset + "_tileDB"):
        ds_tldb = tiledb.open(dataset + "_tileDB", mode="w")
    else:
        y_dim = tiledb.Dim(
            name="y",
            domain=(0, batch_size * num_batches - 1),
            tile=batch_size * num_batches,
            dtype="uint64",
        )
        x_dim = tiledb.Dim(name="x", domain=(0, 784), tile=785, dtype="uint64")
        domain = tiledb.Domain(y_dim, x_dim)
        attr = tiledb.Attr(name="", dtype="int64", var=False)
        schema = tiledb.ArraySchema(
            domain=domain,
            attrs=[attr],
            cell_order="row-major",
            tile_order="row-major",
            sparse=False,
        )
        tiledb.Array.create(dataset + "_tileDB", schema)
        ds_tldb = tiledb.open(dataset + "_tileDB", mode="w")

    assert type(ds_tldb) == tiledb.array.DenseArray
    time_batches(ds_tldb, batch_size, num_batches)


def time_zarr(dataset, batch_size=1, num_batches=1):
    if os.path.exists(dataset + "_zarr"):
        ds_zarr = zarr.open(dataset + "_zarr")
    else:
        ds_zarr = zarr.create(
            shape=(batch_size * num_batches, 785),
            chunks=(batch_size, None),
            store=dataset + "_zarr",
        )

    assert type(ds_zarr) == zarr.core.Array
    time_batches(ds_zarr, batch_size, num_batches)


def time_hub(dataset, batch_size=1, num_batches=1, local=True, user=None):
    my_schema = {
        "image": hub.schema.Image(shape=(28, 28, 1), dtype="uint8"),
        "label": hub.schema.ClassLabel(num_classes=10),
    }
    if local is True:
        ds = hub.Dataset(
            "./" + dataset + "_hub",
            shape=(batch_size * num_batches,),
            schema=my_schema,
            mode="w",
        )
    else:
        ds = hub.Dataset(
            user + "/" + dataset,
            shape=(batch_size * num_batches,),
            schema=my_schema,
            mode="w",
        )

    assert type(ds) == hub.api.dataset.Dataset
    time_batches(ds, batch_size, num_batches, hub=True)


datasets = ["benchmark"]
configs = [
    {"batch_size": 7000, "num_batches": 10},
    {"batch_size": 70000, "num_batches": 1},
]
user = "debadityapal"

if __name__ == "__main__":
    for dataset in datasets:
        for config in configs:
            print(
                "Dataset:",
                dataset,
                "with Batch Size:",
                config["batch_size"],
                "with num_batches:",
                config["num_batches"],
            )
            print("Performance of TileDB")
            time_tiledb(dataset, config["batch_size"], config["num_batches"])
            print("Performance of Zarr")
            time_zarr(dataset, config["batch_size"], config["num_batches"])
            print("Performance of Hub (Stored on the Cloud):")
            time_hub(dataset, config["batch_size"], config["num_batches"], user=user)
            print("Performance of Hub (Stored Locally):")
            time_hub(dataset, config["batch_size"], config["num_batches"], local=True)
