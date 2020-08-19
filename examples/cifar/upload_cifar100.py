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
    classes = {
        "aquatic mammals": ["beaver", "dolphin", "otter", "seal", "whale"],
        "fish": ["aquarium fish", "flatfish", "ray", "shark", "trout"],
        "flowers": ["orchids", "poppies", "roses", "sunflowers", "tulips"],
        "food containers": ["bottles", "bowls", "cans", "cups", "plates"],
        "fruit and vegetables": [
            "apples",
            "mushrooms",
            "oranges",
            "pears",
            "sweet peppers",
        ],
        "household electrical devices": [
            "clock",
            "computer keyboard",
            "lamp",
            "telephone",
            "television",
        ],
        "household furniture": ["bed", "chair", "couch", "table", "wardrobe"],
        "insects": ["bee", "beetle", "butterfly", "caterpillar", "cockroach"],
        "large carnivores": ["bear", "leopard", "lion", "tiger", "wolf"],
        "large man-made outdoor things": [
            "bridge",
            "castle",
            "house",
            "road",
            "skyscraper",
        ],
        "large natural outdoor scenes": ["cloud", "forest", "mountain", "plain", "sea"],
        "large omnivores and herbivores": [
            "camel",
            "cattle",
            "chimpanzee",
            "elephant",
            "kangaroo",
        ],
        "medium-sized mammals": ["fox", "porcupine", "possum", "raccoon", "skunk"],
        "non-insect invertebrates": ["crab", "lobster", "snail", "spider", "worm"],
        "people": ["baby", "boy", "girl", "man", "woman"],
        "reptiles": ["crocodile", "dinosaur", "lizard", "snake", "turtle"],
        "small mammals": ["hamster", "mouse", "rabbit", "shrew", "squirrel"],
        "trees": ["maple", "oak", "palm", "pine", "willow"],
        "vehicles 1": ["bicycle", "bus", "motorcycle", "pickup truck", "train"],
        "vehicles 2": ["lawn-mower", "rocket", "streetcar", "tank", "tractor"],
    }

    superclasses = list(classes.keys())
    subclasses = [item for key in superclasses for item in classes[key]]

    fine_labels = np.concatenate(
        [np.array(d[b"fine_labels"], dtype="int16") for d in dicts]
    )
    coarse_labels = np.concatenate(
        [np.array(d[b"coarse_labels"], dtype="int16") for d in dicts]
    )

    print(images.shape, fine_labels.shape, coarse_labels.shape)
    Image.fromarray(images[1000].transpose(1, 2, 0)).save("./data/image.png")

    images_t = tensor.from_array(images, dtag="image")
    fine_labels_t = tensor.from_array(fine_labels)
    coarse_labels_t = tensor.from_array(coarse_labels)
    classes_t = tensor.from_array(
        np.array([subclasses[label] for label in fine_labels], dtype="U64"),
        dtag="text",
    )
    superclasses_t = tensor.from_array(
        np.array([superclasses[label] for label in coarse_labels], dtype="U64"),
        dtag="text",
    )
    ds = dataset.from_tensors(
        {
            "data": images_t,
            "fine_labels": fine_labels_t,
            "coarse_labels": coarse_labels_t,
            "classes": classes_t,
            "superclasses": superclasses_t,
        }
    )
    ds.store(f"{args.output_name}")


if __name__ == "__main__":
    main()
