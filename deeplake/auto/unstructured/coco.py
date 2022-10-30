import json
import os
import numpy as np
import deeplake

from pathlib import Path
from typing import List, Union
from tqdm import tqdm

from deeplake.core.dataset.dataset import Dataset
from deeplake.htype import HTYPE_SUPPORTED_COMPRESSIONS

from .base import UnstructuredDataset

GENERIC_TENSOR_CONFIG = {"htype": "generic", "sample_compression": None}
IMAGE_TENSOR_CONFIG = {"htype": "image", "sample_compression": "jpeg"}

TENSOR_SETTINGS_CONFIG = {
    "segmentation": {
        "htype": "polygon",
        "sample_compression": None,
    },
    "category_id": {"htype": "class_label", "sample_compression": None},
    "bbox": {"htype": "bbox", "sample_compression": None},
    "_other": GENERIC_TENSOR_CONFIG,
}


def coco_2_deeplake(coco_key, value, tensor_meta, category_lookup=None):
    """Takes a key-value pair from coco data and converts is to data in Deep Lake format
    as per the key types in coco and array shape rules in Deep Lake"""

    dtype = tensor_meta.dtype

    if coco_key == "bbox":
        return np.array(value).astype(dtype)
    elif coco_key == "segmentation":
        # Make sure there aren't multiple segementations per single value, because multiple things will break
        if len(value) > 1:
            print("MULTIPLE SEGMENTATIONS PER OBJECT")

        return np.array(value[0]).reshape(((int(len(value[0]) / 2)), 2)).astype(dtype)

    elif coco_key == "category_id":
        if category_lookup is None:
            return value
        else:
            return category_lookup[str(value)]

    elif coco_key == "keypoints":
        return np.array(value).astype(dtype)


class CocoDataset(UnstructuredDataset):
    @staticmethod
    def _get_annotations(path: str):
        with open(path, "r") as f:
            anns = json.load(f)

        assert "annotations" in anns

        return anns

    def __init__(
        self,
        source: str,
        annotation_files: Union[str, List[str]],
        key_to_tensor_mapping: dict = {},
        file_to_group_mapping: dict = {},
        ignore_one_group: bool = False,
        ignore_keys: Union[str, List[str]] = [],
        image_settings: dict = {},
    ):
        """
        Args:
            source (str): The path to the directory containing images.
            annotation_files (Union[str, List[str]]): The path(s) to the annotation jsons.
            key_to_tensor_mapping (dict): The names to which the keys in the annotation json should be mapped to when creating tensors.
            file_to_group_mapping (dict): Map the annotation file names to groups.
            ignore_one_group (bool): If there is only a single annotation file, whether the creation of group should be skipped.
            ignore_keys (bool): Which keys in the annotation file should be ignored and tensors/data should not be created.
        """

        super().__init__(source)

        self.annotation_files = (
            [annotation_files]
            if not isinstance(annotation_files, list)
            else annotation_files
        )
        self.ignore_one_group = ignore_one_group

        self.key_to_tensor_mapping = key_to_tensor_mapping
        self.file_to_group_mapping = file_to_group_mapping
        self.ignore_keys = ignore_keys
        self.image_settings = image_settings

        self._validate_key_mapping()
        self._validate_group_mapping()

    def _validate_key_mapping(self):
        # Make sure it maps to unique tensors
        assert len(self.key_to_tensor_mapping.values()) == len(
            set(self.key_to_tensor_mapping.values())
        )

    def _validate_group_mapping(self):
        # Make sure it maps to unique groups
        assert len(self.file_to_group_mapping.values()) == len(
            set(self.file_to_group_mapping.values())
        )

    def _parse_tensors(
        self,
        inspect_limit: int = 1000000,
    ) -> ParsedTensorStructure:
        """Return all the tensors and groups that should be created for this dataset"""

        img_config = IMAGE_TENSOR_CONFIG.copy()

        # Change the htype if linked tensors are specified
        if "link" in self.image_settings.keys() and self.image_settings["link"]:
            img_config["htype"] = "link[image]"

        img_config["sample_compression"] = self.image_settings["sample_compression"]

        parsed_structure = [
            {"name": "images", "type": "tensor", "primary": True, "params": img_config}
        ]

        # Iterate through each annotation file and inspect the keys to get Deep Lake tensor structure
        for ann_file in self.annotation_files:
            annotations = CocoDataset._get_annotations(ann_file)
            file_name = Path(ann_file).stem

            # Find all the keys in that annotation file
            keys_in_group = []
            for ann in annotations["annotations"][:inspect_limit]:
                for key in ann.keys():
                    if key not in keys_in_group and key not in self.ignore_keys:
                        keys_in_group.append(key)

            # Create a list of tensors and their properties
            tensor_list = []
            for key in keys_in_group:
                tensor_list.append(
                    {
                        "name": self.key_to_tensor_mapping.get(key, key),
                        "coco_key": key,
                        "type": "tensor",
                        "params": TENSOR_SETTINGS_CONFIG.get(
                            key, GENERIC_TENSOR_CONFIG
                        ),
                    }
                )

            # assert file_name not in parsed_structure.keys()

            parsed_structure.append(
                {
                    "type": "group",
                    "ann_file_path": ann_file,
                    "structure": tensor_list,
                    "ignore": False,
                    "name": self.file_to_group_mapping.get(file_name, file_name),
                }
            )

        if self.ignore_one_group and len(self.annotation_files) == 1:
            parsed_structure[-1]["ignore"] = True

        return parsed_structure

    def _create_tensor(self, ds: Dataset, tensor_structure: dict):
        ds.create_tensor(tensor_structure["name"], **tensor_structure["params"])

    def _parse_dataset(self, ds: Dataset, parsed_structure: dict):
        """Recursively creates groups and tensors from the parsed_structure dictionary"""
        if isinstance(parsed_structure, dict) and parsed_structure["type"] == "tensor":
            # self._create_tensor(ds, parsed_structure)
            ds.create_tensor(parsed_structure["name"], **parsed_structure["params"])

        else:
            for ps in parsed_structure:
                if isinstance(ps, dict) and ps["type"] == "tensor":
                    ds.create_tensor(ps["name"], **ps["params"])
                else:
                    if not ps["ignore"]:
                        ds.create_group(ps["name"])
                        self._parse_dataset(ds[ps["name"]], ps["structure"])
                    else:
                        self._parse_dataset(ds, ps["structure"])

    def _get_image_files(self):
        """Returns a list of image files that can be appended to Deeplake, a list of invalid image files that cannot be appended, the number of different compressions, and the most common compression."""
        # Create a list of valid image files
        img_files = []
        img_file_extentions = []
        invalid_image_files = []
        for file in os.listdir(self.source):
            if file.endswith(tuple(HTYPE_SUPPORTED_COMPRESSIONS["image"] + ["jpg"])):
                img_files.append(file)
                img_file_extentions.append(Path(file).suffix.replace(".", ""))
            else:
                invalid_image_files.append(file)

        unique_file_extentions = set(img_file_extentions)

        return (
            img_files,
            invalid_image_files,
            len(unique_file_extentions),
            max(unique_file_extentions, key=img_file_extentions.count),
        )

    def structure(self, ds: Dataset, use_progress_bar: bool = True):

        # Create a list of valid image files
        (
            img_files,
            invalid_image_files,
            num_compressions,
            most_common_compression,
        ) = self._get_image_files()

        if "sample_compression" not in self.image_settings.keys():
            self.image_settings["sample_compression"] = most_common_compression

        # Parse the dataset structure based on the user inputs and the input data
        parsed = self._parse_tensors()

        # Populate the tensors and groups based on the dataset structure
        self._parse_dataset(ds, parsed_structure=parsed)

        with ds:
            for ann_file in self.annotation_files:
                all_anns = CocoDataset._get_annotations(ann_file)

                # Logic is: iterate over images in our images folder -> for each image pull the the image id -> for each id pull the annotations -> parse data -> append data

                # Create a list of all the image filenames in the annotations file, so we can quickly pull the id corresponding to each image in our input files
                img_files_anns = [item["file_name"] for item in all_anns["images"]]
                # Create a list of ids for each annotation, so we can quickly find the annotations by id
                img_id_anns = [item["image_id"] for item in all_anns["annotations"]]

                # Create an easy mapping to lookup the text label from the coco category id
                id_2_label_mapping = {}
                for item in all_anns["categories"]:
                    id_2_label_mapping[str(item["id"])] = item["name"]

                # Though logically less ideal, we have to iterate over the images, beucase there are multiple annotations per image, and thus mutliple annotations per row in the Deep Lake dataset

                for img_file in tqdm(img_files):
                    try:
                        index = img_files_anns.index(img_file)
                        img_id = all_anns["images"][index]["id"]
                    except Exception as e:
                        print(str(e))

                    # Find the annotations that match the image id
                    matching_anns = [
                        all_anns["annotations"][i]
                        for i, id in enumerate(img_id_anns)
                        if id == img_id
                    ]

                    # Find which group has this file
                    group = [
                        s
                        for s in parsed
                        if s["type"] == "group" and s["ann_file_path"] == ann_file
                    ][0]

                    tensors = group["structure"]
                    values = [
                        [] for _ in range(len(tensors))
                    ]  # Todo: Initialize empty arrays instead of lists. This should be much faster.

                    # Create the objects to which data will be appended. We need to know if it's a
                    if "ignore" in group.keys() and group["ignore"]:
                        append_obj = ds
                    else:
                        append_obj = ds[group["name"]]

                    # Create a list of lists with all the data
                    for ann in matching_anns:

                        for i, tensor in enumerate(tensors):

                            coco_key = tensor["coco_key"]
                            value = coco_2_deeplake(
                                coco_key,
                                ann[coco_key],
                                append_obj[tensor["name"]].meta,
                                category_lookup=id_2_label_mapping,
                            )

                            values[i].append(value)

                    # Append the annotation data
                    for i, tensor in enumerate(tensors):
                        append_obj[tensor["name"]].append(values[i])

            primary_tensor = [
                structure
                for structure in parsed
                if "primary" in structure.keys()
                and structure["primary"] == True
                and structure["type"] == "tensor"
            ]

            assert len(primary_tensor) == 1

            for img_file in img_files:
                # Append the image data
                ds[primary_tensor[0]["name"]].append(
                    deeplake.read(os.path.join(self.source, img_file))
                )
