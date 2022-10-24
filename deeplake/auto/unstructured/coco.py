import json
import os
import numpy as np

from pathlib import Path
from typing import List, Union
from tqdm import tqdm

from deeplake.core.dataset.dataset import Dataset
from deeplake.util.convert import ParsedTensorStructure, coco_2_deeplake

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
    ) -> ParsedTensorStructure:
        """Return all the tensors and groups that should be created for this dataset"""
        parsed_structure = {}
        parsed_structure = ParsedTensorStructure(
            groups={}, tensor_settings=TENSOR_SETTINGS_CONFIG, ignore_one_group=False
        )

        # Iterate through each annotation file and inspect the keys to get Deep Lake tensor structure
        for ann_file in self.annotation_files:
            annotations = CocoDataset._get_annotations(ann_file)
            file_name = Path(ann_file).name

            keys_in_group = set()
            for ann in annotations["annotations"][:inspect_limit]:
                keys_in_group.update(ann.keys())

            group_name = self.file_to_group_mapping.get(file_name, file_name)

            parsed_structure.set_group(
                group_name,
                structure={
                    "file_path": ann_file,
                    "raw_keys": list(keys_in_group),
                    "tensor_names": [
                        self.key_to_tensor_mapping.get(k, k) for k in keys_in_group
                    ],
                },
            )

        return parsed_structure

    def structure(self, ds: Dataset, use_progress_bar: bool = True):
        img_files = os.listdir(self.source)
        parsed_structure = self._parse_tensors()

        parsed_structure.create_tensors(ds)
        parsed = parsed_structure.groups

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
                for img_file in tqdm(img_files, disable=not use_progress_bar):
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
                        k for k, v in parsed.items() if v.file_path == ann_file
                    ][0]

                    tensors = parsed[group_name].raw_keys
                    values = [[] for _ in range(len(tensors))]

                    # Create the objects to which data will be appended. We need to know if it's a
                    append_obj = ds[group_name]

                    # Create a list of lists with all the data
                    for ann in matching_anns:
                        for i, key in enumerate(parsed[group_name].raw_keys):

                            value = coco_2_deeplake(
                                key,
                                ann[key],
                                append_obj[
                                    parsed[group_name].tensor_names[i]
                                ].meta.dtype,
                                category_lookup=id_2_label_mapping,
                            )

                            values[i].append(value)

                    # Append the data
                    tensor_names = parsed[group_name].tensor_names
                    for i, tensor_name in enumerate(tensor_names):
                        append_obj[tensor_names[i]].append(values[i])
