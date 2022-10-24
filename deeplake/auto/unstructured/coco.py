import json
import os
import numpy as np

from pathlib import Path
from typing import List, Union
from tqdm import tqdm

from deeplake.core.dataset.dataset import Dataset


from .base import UnstructuredDataset

GENERIC_TENSOR_CONFIG = {"htype": "generic", "sample_compression": "lz4"}

TENSOR_SETTINGS_CONFIG = {
    "segmentation": {
        "htype": "polygon",
        "sample_compression": "lz4",
    },
    "category_id": {"htype": "class_label", "sample_compression": "lz4"},
    "bbox": {"htype": "bbox", "sample_compression": "lz4"},
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

    else:
        return value


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
    ):
        """
        Args:
            source (str): The path to the directory containing images.
            annotation_files (Union[str, List[str]]): The path(s) to the annotation jsons.
            key_to_tensor_mapping (dict): The names to which the keys in the annotation json should be mapped to when creating tensors.
            file_to_group_mapping (dict): Map the annotation file names to groups.
            ignore_one_group (bool): If there is only a single annotation file, wether the creation of group should be skipped.
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
    ) -> dict:
        """Return all the tensors and groups that should be created for this dataset"""
        parsed_structure = {}

        # Iterate through each annotation file and inspect the keys to back our the tensors
        for ann_file in self.annotation_files:
            annotations = CocoDataset._get_annotations(ann_file)
            file_name = Path(ann_file).stem

            keys_in_group = []
            for ann in annotations["annotations"][:inspect_limit]:
                for key in ann.keys():
                    if key not in keys_in_group:
                        keys_in_group.append(key)

            group_name = self.file_to_group_mapping.get(file_name, file_name)

            parsed_structure[group_name] = {
                "file_path": ann_file,
                "raw_keys": keys_in_group,
                'ignore_group': False,
                "renamed_tensors": [
                    self.key_to_tensor_mapping.get(k, k) for k in keys_in_group
                ],
            }

        return parsed_structure

    def _create_tensors(self, ds: Dataset, parsed_structure: dict):
        for i, raw_key in enumerate(parsed_structure["raw_keys"]):
            tensor_settings = TENSOR_SETTINGS_CONFIG.get(raw_key, GENERIC_TENSOR_CONFIG)
            ds.create_tensor(parsed_structure["renamed_tensors"][i], **tensor_settings)

    def _create_groups(self, ds: Dataset, parsed_structure: dict):
        
        for key in parsed_structure.keys():
            if parsed_structure[key]['ignore_group']:
                self._create_tensors(ds, parsed_structure[key]) 
            else:
                ds.create_group(key)
                self._create_tensors(ds[key], parsed_structure[key])

    def structure(self, ds: Dataset, use_progress_bar: bool = True):
        img_files = os.listdir(self.source)
        parsed = self._parse_tensors()

        self._create_groups(ds, parsed_structure=parsed)

        if self.ignore_one_group and len(parsed.keys())==1: parsed[list(parsed.keys())[0]]['ignore_group'] = self.ignore_one_group

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
                    group_name = [
                        k for k, v in parsed.items() if v["file_path"] == ann_file
                    ][0]

                    tensors = parsed[group_name]["raw_keys"]
                    values = [[] for _ in range(len(tensors))]

                    # Create the objects to which data will be appended. We need to know if it's a
                    if (
                        "ignore" in parsed[group_name].keys()
                        and parsed[group_name]["ignore"]
                    ):
                        append_obj = ds
                    else:
                        append_obj = ds[group_name]

                    # Create a list of lists with all the data
                    for ann in matching_anns:
                        for i, key in enumerate(parsed[group_name]["raw_keys"]):

                            value = coco_2_deeplake(
                                key,
                                ann[key],
                                append_obj[
                                    parsed[group_name]["renamed_tensors"][i]
                                ].meta,
                                category_lookup=id_2_label_mapping,
                            )

                            values[i].append(value)

                    # Append the data
                    tensor_names = parsed[group_name]["renamed_tensors"]
                    for i, tensor_name in enumerate(tensor_names):
                        append_obj[tensor_names[i]].append(values[i])
