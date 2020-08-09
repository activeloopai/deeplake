import numpy as np

import hub
import tensorflow as tf
import torch
from torch.utils.data import Dataset, DataLoader


def main():
    tf.enable_eager_execution()
    t1 = hub.tensor.from_array(np.array([1, 2, 3]))
    t2 = hub.tensor.from_array(np.array([4, 5, 6]))
    ds = hub.dataset.from_tensors({"first": t1, "second": t2})
    ds = ds.to_tensorflow()
    for x in ds:
        print(x)


def main2():
    tf.enable_eager_execution()
    ds = hub.load("s3://snark-hub/public/coco/coco2017")
    print(ds["id"][0].compute())
    exit()
    # ds = hub.dataset.from_tensors(
    #     {
    #         "image_id": ds["image_id"],
    #         "image": ds["image"],
    #         "category_id": ds["category_id"],
    #     }
    # )
    # print("********************", ds["area"][0].compute())
    # for i in range(len(ds)):
    #     item = ds[i]
    #     print(item)
    #     for key, value in item.items():
    #         print(key, value)
    #         print(key, value.compute())
    #     break

    # ds = hub.load("s3://snark-hub/public/cifar/cifar10")
    # ds = hub.load("s3://snark-hub/public/mnist/mnist")
    # ds = ds.to_tensorflow()
    ds = ds.to_pytorch()
    ds = DataLoader(ds, num_workers=0, collate_fn=ds.collate_fn, batch_size=1)
    for x in ds:
        # print(len(x))
        sample = x
        print(sample.keys())
        print(sample["image_id"])
        break


if __name__ == "__main__":
    main2()
