import argparse
import os
import pickle
import numpy as np
from fnmatch import fnmatch
from PIL import Image

from hub.collections import dataset, tensor

import torch, torchvision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        help="Path to an original imagenet dataset",
        default="./data/imagenet",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Path to a transformed dataset",
        default="imagenet",
    )
    args = parser.parse_args()
    files = sorted([f for f in os.listdir(args.input) if "_batch" in f])
    
    dicts = []
    for f in files:
        with open(os.path.join(args.input, f), "rb") as fh:
            dicts += [pickle.load(fh, encoding="bytes")]
            print(dicts[-1]['mean'])

    images = np.concatenate([d["data"] for d in dicts])
    images = images.reshape((len(images), 3, 8, 8))
    labels = np.concatenate([np.array(d["labels"], dtype="int16") for d in dicts])
    print(images.shape, labels.shape)
    images_t = tensor.from_array(images)
    labels_t = tensor.from_array(labels)
    ds = dataset.from_tensors({"data": images_t, "labels": labels_t})
    ds.store(f"{args.output}")


if __name__ == "__main__":
    main()
