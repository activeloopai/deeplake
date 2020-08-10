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
        default="./data/cifar10",
    )
    parser.add_argument(
        "output_name",
        metavar="N",
        type=str,
        help="Dataset output name",
        default="cifar10",
    )
    args = parser.parse_args()
    files = sorted([f for f in os.listdir(args.dataset_path) if "_batch" in f])
    dicts = []
    for f in files:
        with open(os.path.join(args.dataset_path, f), "rb") as fh:
            dicts += [pickle.load(fh, encoding="bytes")]
            print(dicts[-1].keys())
    images = np.concatenate([d[b"data"] for d in dicts])
    images = images.reshape((len(images), 3, 32, 32))
    labels = np.concatenate([np.array(d[b"labels"], dtype="int16") for d in dicts])
    print(images.shape, labels.shape)
    Image.fromarray(images[1000].transpose(1, 2, 0)).save("./data/image.png")
    images_t = tensor.from_array(images)
    labels_t = tensor.from_array(labels)
    ds = dataset.from_tensors({"data": images_t, "labels": labels_t})
    ds.store(f"{args.output_name}")


if __name__ == "__main__":
    main()
