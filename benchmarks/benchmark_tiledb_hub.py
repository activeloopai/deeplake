"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import tiledb
import hub
import numpy as np
import os
from time import time
from hub.utils import Timer


def time_tiledb(dataset, batch_size=1):
    ds = hub.Dataset(dataset, cache=False, storage_cache=False, mode="r")
    if os.path.exists(dataset.split("/")[1] + "_tileDB"):
        ds_tldb = tiledb.open(dataset.split("/")[1] + "_tileDB")
    else:
        if not os.path.exists(dataset.split("/")[1] + "_tileDB"):
            os.makedirs(dataset.split("/")[1] + "_tileDB")
        ds_numpy = np.concatenate(
            (
                ds["image"].compute().reshape(ds.shape[0], -1),
                ds["label"].compute().reshape(ds.shape[0], -1),
            ),
            axis=1,
        )
        ds_tldb = tiledb.from_numpy(dataset.split("/")[1] + "_tileDB", ds_numpy)

    assert type(ds_tldb) == tiledb.array.DenseArray

    with Timer("Time"):
        counter = 0
        t0 = time()
        for batch in range(ds.shape[0] // batch_size):
            x, y = (
                ds_tldb[batch * batch_size : (batch + 1) * batch_size, :-1],
                ds_tldb[batch * batch_size : (batch + 1) * batch_size, -1],
            )
            counter += 1
            t1 = time()
            print("Batch", counter, f"dt: {t1 - t0}")
            t0 = t1


def time_hub(dataset, batch_size=1):
    ds = hub.Dataset(dataset, cache=False, storage_cache=False, mode="r")

    assert type(ds) == hub.api.dataset.Dataset

    with Timer("Time"):
        counter = 0
        t0 = time()
        for batch in range(ds.shape[0] // batch_size):
            x, y = (
                ds[batch * batch_size : (batch + 1) * batch_size]["image"].compute(),
                ds[batch * batch_size : (batch + 1) * batch_size]["label"].compute(),
            )
            counter += 1
            t1 = time()
            print("Batch", counter, f"dt: {t1 - t0}")
            t0 = t1


datasets = ["activeloop/mnist"]
batch_sizes = [70000, 7000]


if __name__ == "__main__":
    for dataset in datasets:
        data = hub.Dataset.from_tfds(dataset.split("/")[1])
        data.store("./" + dataset.split("/")[1] + "_hub")

        for batch_size in batch_sizes:
            print("Dataset: ", dataset, "with Batch Size: ", batch_size)
            print("Performance of TileDB")
            time_tiledb(dataset, batch_size)
            print("Performance of Hub (Stored on the Cloud):")
            time_hub(dataset, batch_size)
            print("Performance of Hub (Stored Locally):")
            time_hub("./" + dataset.split("/")[1] + "_hub", batch_size)
