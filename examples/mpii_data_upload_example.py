import json
import time
import numpy as np
from PIL import Image
from tqdm import tqdm

import hub
from hub import Dataset, schema
from hub.schema import Tensor, Text

"""
Below we will define a schema for our dataset. Schema is kind of
a container to specify structure, shape, dtype and meta information
of our dataset. We have different types of schemas for different
types of data like image, tensor, text. More info. on docs.
"""
mpii_schema = {
    """
    we specify 'shape' as None for variable image size, and we
    give 'max_shape' arguement a maximum possible size of image.
    """
    "image": schema.Image(
        shape=(None, None, 3),
        max_shape=(1920, 1920, 3),
        dtype="uint8"),
    "isValidation": "float64",
    "img_paths": Text(shape = (None,), max_shape=(15,)),
    "img_width": "int32",
    "img_height": "int32",
    "objpos": Tensor(max_shape=(100,), dtype="float64"),
    """
    'joint_self' has nested list structure
    """
    "joint_self": Tensor(
        shape=(None, None),
        max_shape=(100, 100),
        dtype="float64"),
    "scale_provided": "float64",
    "annolist_index": "int32",
    "people_index": "int32",
    "numOtherPeople": "int32",
}


"""
Below function takes JSON file and gives annotations in the
form of dictionary inside list.
"""
def get_anno(jsonfile):

    with open(jsonfile) as f:
        instances = json.load(f)

    annotations = []
    for i in range(len(instances)):
        annotations.append(instances[i])
    return annotations


"""
Hub Transform is optimized to give efficient processing and
storing of dataset. Below function takes a dataset and applies
transform on every sample(instance) of dataset, and outputs a
dataset with specified schema. More info. on docs.
"""
@hub.transform(schema = my_schema, workers= 8)
def my_transform(annotation):
    return{
    "image": np.array(Image.open(img_path + annotation["img_paths"])),
    "isValidation": np.array(annotation["isValidation"]),
    "img_paths": annotation["img_paths"],
    "img_width": np.array(annotation["img_width"]),
    "img_height": np.array(annotation["img_height"]),
    "objpos": np.array(annotation["objpos"]),
    "joint_self":  np.array(annotation["joint_self"]),
    "scale_provided": np.array(annotation["scale_provided"]),
    "annolist_index": np.array(annotation["annolist_index"]),
    "people_index": np.array(annotation["people_index"]),
    "numOtherPeople": np.array(annotation["numOtherPeople"]),
    }


if __name__ == "__main__":

    tag = input("Enter tag(username/dataset_name):")
    jsonfile = input("Enter json file path:")
    img_path = input("Enter path to images:")

    annotations = get_anno(jsonfile)

    t1 = time.time()
    ds = my_transform(annotations)
    ds = ds.store(tag)
    print("Time taken to upload:", (time.time() - t1), "sec")

"""
Dataset uploaded using AWS EC2. Pipeline took 8931.26 sec to
finish. Dataset is visible on app and tested working fine.
"""
