import os
from PIL import Image
from hub import Transform, dataset
import numpy as np
import scipy.io


class CubGenerator(Transform):
    def meta(self):
        return {
            "labels": {"shape": (1,), "dtype": "str", "dtag": "text"},
            "image": {"shape": (1,), "dtype": "object", "dtag": "image"},
            "__header__": {"shape": (1,), "dtype": "str", "dtag": "text"},
            "__version__": {"shape": (1,), "dtype": "str", "dtag": "text"},
            "wikipedia_url": {"shape": (1,), "dtype": "str", "dtag": "text"},
            "seg": {"shape": (1,), "dtype": "int", "dtag": "segmentation"},
            "bbox": {"shape": (1,), "dtype": "int32", "dtag": "box"},
            "flickr_url": {"shape": (1,), "dtype": "str", "dtag": "text"},
        }

    def forward(self, image_info):
        ds = {}
        ds["labels"] = np.empty(1, dtype="str")
        ds["labels"][0] = image_info["label"]

        ds["image"] = np.empty(1, object)
        ds["image"][0] = np.array(Image.open(image_info["image_path"]).convert("RGB"))

        ds["__header__"] = np.empty(1, object)
        ds["__header__"][0] = image_info["anno"]["__header__"]

        ds["__version__"] = np.empty(1, object)
        ds["__version__"][0] = image_info["anno"]["__version__"]

        ds["wikipedia_url"] = np.empty(1, object)
        ds["wikipedia_url"][0] = image_info["anno"]["wikipedia_url"]

        ds["seg"] = np.empty(1, object)
        ds["seg"][0] = image_info["anno"]["seg"]

        ds["bbox"] = np.empty(1, object)
        ds["bbox"][0] = image_info["anno"]["bbox"]

        ds["flickr_url"] = np.empty(1, object)
        ds["flickr_url"][0] = image_info["anno"]["flickr_url"]
        return ds


def load_dataset(path):
    image_info_list = []
    with open(os.path.join(path, "attributes", "images-dirs.txt"), "r") as fr:
        images = [filename.split()[1] for filename in fr.readlines()]

    for image in images:
        image_info = {}
        anno = scipy.io.loadmat(os.path.join(path, "annotations-mat", image[:-4]))

        image_info["anno"] = anno
        image_info["image_path"] = os.path.join(path, "images", image)
        image_info["label"] = image.split("/")[0]
        if os.path.exists(os.path.join(path, "images", image)):
            image_info_list.append(image_info)

    ds = dataset.generate(CubGenerator(), image_info_list)
    return ds


if __name__ == "__main__":
    dataPath = input("Enter path to dataset : ")
    ds = load_dataset(dataPath)
    dataStore = input("Enter username/dataset")
    ds.store(dataStore)
