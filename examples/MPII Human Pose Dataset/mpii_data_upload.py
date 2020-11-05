import os
import json
import time
import numpy as np
from PIL import Image

import hub
from hub.collections import dataset
from hub.log import logger


class MPIIGenerator(dataset.DatasetGenerator):
    """
    Purpose of generator class is to return dictionary of arrays for the individual example.
    This class inherites from dataset.DatasetGenerator so it is called iteratively for the
    number of examples present in dataset.
    We have defined two functions here:
    meta: returns keys of the dictinary which are refered as features and labels of dataset.
    __call__ : this function takes the list of features(annotations in this case)as input and
    adds the features(or labels) according to the keys in a row of dataset.
    """

    def meta(self):

        return {
            "image": {"shape": (1,), "dtype": "object", "chunksize": 5},
            "dataset": {"shape": (1,), "dtype": "object", "chunksize": 5},
            "isValidation": {"shape": (1,), "dtype": "object", "chunksize": 5},
            "img_paths": {"shape": (1,), "dtype": "str", "chunksize": 5},
            "img_width": {"shape": (1,), "dtype": "object", "chunksize": 5},
            "img_height": {"shape": (1,), "dtype": "object", "chunksize": 5},
            "objpos": {"shape": (1,), "dtype": "object", "chunksize": 5},
            "joint_self": {"shape": (1,), "dtype": "object", "chunksize": 5},
            "scale_provided": {"shape": (1,), "dtype": "object", "chunksize": 5},
            "joint_others": {"shape": (1,), "dtype": "object", "chunksize": 5},
            "scale_provided_other": {
                "shape": (1,),
                "dtype": "object",
                "chunksize": 5,
            },
            "objpos_other": {"shape": (1,), "dtype": "object", "chunksize": 5},
            "annolist_index": {"shape": (1,), "dtype": "object", "chunksize": 5},
            "people_index": {"shape": (1,), "dtype": "object", "chunksize": 5},
            "numOtherPeople": {"shape": (1,), "dtype": "object", "chunksize": 5},
        }

    def __call__(self, input):

        try:
            ds = {}
            img_path = "/home/sanchit/images/"

            n = 1  # for 1 row
            ds["image"] = np.empty(n, object)
            ds["dataset"] = np.empty(n, object)
            ds["isValidation"] = np.empty(n, object)
            ds["img_paths"] = np.empty(n, object)
            ds["img_width"] = np.empty(n, object)
            ds["img_height"] = np.empty(n, object)
            ds["objpos"] = np.empty(n, object)
            ds["joint_self"] = np.empty(n, object)
            ds["scale_provided"] = np.empty(n, object)
            ds["joint_others"] = np.empty(n, object)
            ds["scale_provided_other"] = np.empty(n, object)
            ds["objpos_other"] = np.empty(n, object)
            ds["annolist_index"] = np.empty(n, object)
            ds["people_index"] = np.empty(n, object)
            ds["numOtherPeople"] = np.empty(n, object)

            ds["image"][0] = np.array(Image.open(img_path + input["img_paths"]))
            ds["dataset"][0] = input["dataset"]
            ds["isValidation"][0] = input["isValidation"]
            ds["img_paths"][0] = input["img_paths"]
            ds["img_width"][0] = input["img_width"]
            ds["img_height"][0] = input["img_height"]
            """
            Some features in input list has another list(more than one list) inside them.
            So they are converted to array using np.array(list(list()).
            """
            ds["objpos"][0] = np.array(input["objpos"])
            ds["joint_self"][0] = np.array(input["joint_self"])
            ds["scale_provided"][0] = input["scale_provided"]
            ds["joint_others"][0] = np.array(input["joint_others"])
            ds["scale_provided_other"][0] = np.array(input["scale_provided_other"])
            ds["objpos_other"][0] = np.array(input["objpos_other"])
            ds["annolist_index"][0] = input["annolist_index"]
            ds["people_index"][0] = input["people_index"]
            ds["numOtherPeople"][0] = input["numOtherPeople"]

            return ds

        except Exception as e:
            logger.error(e, exc_info=e, stack_info=True)


def load_dataset():
    """
    This function is used to load json annotations in the form of dictionary and then
    appending all the dictionaries(25205 examples) in the list named annotations.
    Then this list is given as input to __call__ function of generator class.
    dataset.generate calls the generator class iteratively for each dictionary present
    in the list(annotations). Finally it returns the complete dataset with all examples.
    """

    with open("/home/sanchit/mpii_annotations.json", "r") as f:

        instances = json.load(f)

    annotations = []

    for i in range(len(instances)):
        annotations.append(instances[i])

    #   print(annotations[:5])
    print("Annotations loaded.")

    ds = dataset.generate(MPIIGenerator(), annotations)
    print("Dataset generated.")

    return ds


t1 = time.time()
# Call the load_dataset function to generate the complete dataset
ds = load_dataset()
# ds.store stores the dataset in username/MPII_human_pose_data
ds.store("MPII_human_pose_data")
t2 = time.time()
print(f"Pipeline took {(t2 - t1) / 60} minutes")
