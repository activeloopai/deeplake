import argparse
import os
import pickle

import numpy as np
from PIL import Image

from hub.collections import dataset, tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_path",
        metavar="P",
        type=str,
        help="Path to cifar dataset",
        default="./data/cifar100",
    )
    parser.add_argument(
        "output_name",
        metavar="N",
        type=str,
        help="Dataset output name",
        default="cifar100",
    )
    args = parser.parse_args()
    files = ["train", "test"]
    dicts = []
    for f in files:
        with open(os.path.join(args.dataset_path, f), "rb") as fh:
            dicts += [pickle.load(fh, encoding="bytes")]
            print(dicts[-1].keys())
    images = np.concatenate([d[b"data"] for d in dicts])
    images = images.reshape((len(images), 3, 32, 32))
    fine_labels = np.concatenate(
        [np.array(d[b"fine_labels"], dtype="int16") for d in dicts]
    )
    coarse_labels = np.concatenate(
        [np.array(d[b"coarse_labels"], dtype="int16") for d in dicts]
    )
    print(images.shape, fine_labels.shape, coarse_labels.shape)
    Image.fromarray(images[1000].transpose(1, 2, 0)).save("./data/image.png")

    images_t = tensor.from_array(images)
    fine_labels_t = tensor.from_array(fine_labels)
    coarse_labels_t = tensor.from_array(coarse_labels)
    ds = dataset.from_tensors(
        {
            "data": images_t,
            "fine_labels": fine_labels_t,
            "coarse_labels": coarse_labels_t,
        }
    )
    ds.store(f"./data/generated/{args.output_name}")


if __name__ == "__main__":
    main()
