import argparse
import os
import struct

import numpy as np
from array import array as pyarray

from hub.collections import dataset, tensor


def load_mnist(dataset="training", digits=np.arange(10), path=".", size=60000):
    if dataset == "training":
        fname_img = os.path.join(path, "train-images-idx3-ubyte")
        fname_lbl = os.path.join(path, "train-labels-idx1-ubyte")
    elif dataset == "testing":
        fname_img = os.path.join(path, "t10k-images-idx3-ubyte")
        fname_lbl = os.path.join(path, "t10k-labels-idx1-ubyte")

    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, "rb")
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, "rb")
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [k for k in range(size) if lbl[k] in digits]
    N = size  # int(len(ind) * size/100.)
    images = np.zeros((N, rows, cols), dtype=np.uint8)
    labels = np.zeros((N, 1), dtype=np.int8)
    for i in range(N):  # int(len(ind) * size/100.)):
        images[i] = np.array(
            img[ind[i] * rows * cols : (ind[i] + 1) * rows * cols]
        ).reshape((rows, cols))
        labels[i] = lbl[ind[i]]
    labels = [label[0] for label in labels]
    return images, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset_path",
        type=str,
        help="Path to cifar dataset",
        default="./data/mnist",
    )
    parser.add_argument(
        "-o",
        "--output_name",
        type=str,
        help="Dataset output name",
        default="mnist",
    )
    args = parser.parse_args()
    files = ["training", "testing"]
    dicts = []
    for f in files:
        images, labels = load_mnist(f, path=args.dataset_path)
        dicts += [{"images": images, "labels": labels}]
    images = np.concatenate([d["images"] for d in dicts])
    # images = images.reshape((len(images), 3, 32, 32))
    labels = np.concatenate([np.array(d["labels"], dtype="int8") for d in dicts])
    print(images.shape, labels.shape)
    # Image.fromarray(images[1000]).save("./data/image.png")
    images_t = tensor.from_array(images, dtag="mask")
    labels_t = tensor.from_array(labels)
    # coarse_labels_t = tensor.from_array(coarse_labels)
    ds = dataset.from_tensors({"data": images_t, "labels": labels_t})
    ds.store(f"{args.output_name}")


if __name__ == "__main__":
    main()
