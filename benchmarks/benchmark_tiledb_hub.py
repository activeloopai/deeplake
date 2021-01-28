import tiledb
import hub
import numpy as np
import os
from time import time
from hub.utils import Timer


def time_tiledb(dataset, batch_size=1):
    ds = hub.Dataset(dataset)
    if os.path.exists("./test/" + dataset.split("/")[1]):
        ds_tldb = tiledb.open("./test/" + dataset.split("/")[1])
    else:
        if not os.path.exists("./test"):
            os.makedirs("test")
        ds_numpy = np.concatenate(
            (
                ds["image"].compute().reshape(ds.shape[0], -1),
                ds["label"].compute().reshape(ds.shape[0], -1),
            ),
            axis=1,
        )
        ds_tldb = tiledb.from_numpy("./test/" + dataset.split("/")[1], ds_numpy)

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
    ds = hub.Dataset(dataset)

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
        for batch_size in batch_sizes:
            print("Dataset: ", dataset, "with Batch Size: ", batch_size)
            print("Performance of TileDB")
            time_tiledb(dataset, batch_size)
            print("Performance of Hub")
            time_hub(dataset, batch_size)
