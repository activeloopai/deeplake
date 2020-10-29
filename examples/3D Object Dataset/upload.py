"""
Dataset Download Source: http://cvgl.stanford.edu/data2/3Ddataset.zip
Dataset format: Images(.bmp file)
Dataset Features: bicycle, car, cellphone, head, iron, monitor, mouse, shoe, stapler, toaster

Folder Structure:
3Ddataset
  -bicycle
    -bicycle_1
      - Various Images in .bmp format
    -bicycle_2
    -bicycle_3
    ...
    -bicycle_10
  -car
    -car_1
    -car_2
    ...
    -car_10
  ...
  Total 10 features
"""

import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from hub import Transform, dataset
import pandas as pd

NUM_FEATURES = 10


class DatasetGenerator(Transform):
    def meta(self):
        # here we specify the attributes of return type
        return {
            "image_label": {"shape": (1,), "dtype": "int", "dtag": "text"},
            "named_image_label": {"shape": (1,), "dtype": "U25", "dtag": "text"},
            "image": {
                "shape": (1,),
                "dtype": "uint32",
                "chunksize": 100,
                "dtag": "image",
            },
        }

    def forward(self, image_info):
        # we need to return a dictionary of numpy arrays from here
        ds = {}
        ds["image_label"] = np.empty(1, dtype="int")
        ds["image_label"][0] = image_info["image_label"]

        ds["named_image_label"] = np.empty(1, dtype="object")
        ds["named_image_label"][0] = image_info["named_image_label"]

        ds["image"] = np.empty(1, object)
        ds["image"][0] = np.array(Image.open(image_info["image_path"]).convert("RGB"))
        print("------------------------------------------------")
        print(ds["named_image_label"][0] + " image loaded successfully")
        return ds


def map_labels(labels_list):
    dic = {labels_list[i]: i for i in range(1, NUM_FEATURES + 1)}
    return dic


def load_dataset(base_path):
    labels_list = os.listdir(base_path)
    labels_dict = map_labels(labels_list)
    image_info_list = []
    for label in labels_list:
        for label_num in range(1, NUM_FEATURES + 1):
            curr_path = base_path + "/" + label + "/" + label + "_" + str(label_num)
            images_list = os.listdir(curr_path)
            for image in images_list:
                image_info = {}
                if image.lower().startswith(
                    label
                ):  # all images' name starts with the feature name (observation)
                    image_info["image_path"] = curr_path + "/" + image
                    image_info["image_label"] = labels_dict[label]
                    image_info["named_image_label"] = label
                    image_info_list.append(image_info)

    # the generator iterates through the argument given, one by one and applies forward. This is done lazily.
    ds = dataset.generate(DatasetGenerator(), image_info_list)
    return ds


def main():
    base_path = "./3Ddataset"
    # stores the dataset in username/datasetname
    ds = load_dataset(base_path)
    ds.store("ThreeDimensionalDataset")


if __name__ == "__main__":
    main()
