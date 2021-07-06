import numpy as np
import logging

logging.basicConfig(level=logging.ERROR)

import time
import hub
from benchmark_utils import timer, parametrize
from tfds_utils import datasets, download, DATASETS_DIR, get_hub_ds_path
import posixpath


username = open("username.txt", "r").read()


@parametrize(dataset=datasets, use_transform=[True, False])
def upload_local(dataset, use_transform):
    ds_local = download(dataset)
    cloud_path = f"hub://{username}/{dataset}"
    ds_cloud = hub.Dataset(cloud_path)
    hub.Dataset.delete(ds_cloud)
    ds_cloud = hub.Dataset(cloud_path)
    for tensor in ds_local.tensors:
        ds_cloud.create_tensor(tensor)
    with timer("upload_local.write", config={"dataset": dataset, "use_transform": use_transform}):
        with ds_cloud:
            if use_transform:
                hub.transform(ds_local, [lambda x: x], ds_cloud)
            else:
                for tensor in ds_local.tensors:
                    ds_cloud[tensor].extend(ds_local[tensor].numpy(aslist=True))
    ds_cloud.clear_cache()
    arrays = []
    with timer("upload_local.read", config={"dataset": dataset}):
        for tensor in ds_cloud.tensors:
            arrays[tensor] = ds_cloud[tensor].numpy(aslist=True)
    for k, v in arrays:
        arrs1 = arrays[k]
        arrs2 = ds_local[k].numpy(aslist=True)
        for arr1, arr2 in zip(arrs1, arrs2):
            np.testing.assert_array_equal(arr1, arr2)


if __name__ == "__main__":
    upload_local()
