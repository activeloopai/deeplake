import os
import deeplake

from pathlib import Path
from typing import List, Union, Dict
from tqdm import tqdm
from itertools import chain

from deeplake.core.dataset.dataset import Dataset
from deeplake.util.exceptions import IngestionError
from deeplake.client.log import logger

from ..base import UnstructuredDataset
from ..util import DatasetStructure, GroupStructure, TensorStructure
from .convert import coco_2_deeplake, CocoAnnotation, CocoImages

from .constants import (
    DEFAULT_GENERIC_TENSOR_PARAMS,
    DEFAULT_COCO_TENSOR_PARAMS,
    DEFAULT_IMAGE_TENSOR_PARAMS,
)


class CocoDataset(UnstructuredDataset):
    def __init__(
        self,
        source: str,
        annotation_files: Union[str, List[str]],
        key_to_tensor_mapping: Dict = {},
        file_to_group_mapping: Dict = {},
        ignore_one_group: bool = False,
        ignore_keys: Union[str, List[str]] = [],
        image_settings: Dict = {},
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
        self.images = CocoImages(images_directory=source)

        self.annotation_files = (
            [annotation_files]
            if not isinstance(annotation_files, list)
            else annotation_files
        )
        self.ignore_one_group = ignore_one_group

        self.key_to_tensor = key_to_tensor_mapping
        self._validate_key_mapping()
        self.tensor_to_key = {v: k for k, v in key_to_tensor_mapping.items()}

        self.file_to_group = {Path(k).stem: v for k, v in file_to_group_mapping.items()}
        self._validate_group_mapping()

        self.ignore_keys = ignore_keys
        self.image_settings = image_settings

    def _validate_key_mapping(self):
        if len(self.key_to_tensor.values()) != len(set(self.key_to_tensor.values())):
            raise IngestionError("Keys must be mapped to unique tensor names.")

    def _validate_group_mapping(self):
        if len(self.file_to_group.values()) != len(set(self.file_to_group.values())):
            raise IngestionError("File names must be mapped to unique group names.")

    def _parse_annotation_tensors(
        self,
        inspect_limit: int = 1000000,
    ) -> DatasetStructure:
        """Return all the tensors and groups that should be created for this dataset"""
        dataset_structure = DatasetStructure(ignore_one_group=self.ignore_one_group)

        for ann_file in self.annotation_files:
            coco_file = CocoAnnotation(file_path=ann_file)
            annotations = coco_file.annotations
            file_name = Path(ann_file).stem
            keys_in_group = set(chain.from_iterable(annotations[:inspect_limit]))

            group = GroupStructure(
                self.file_to_group.get(file_name, file_name),
            )

            for key in keys_in_group:
                if key in self.ignore_keys:
                    continue

                tensor = TensorStructure(
                    name=self.key_to_tensor.get(key, key),
                    params=DEFAULT_COCO_TENSOR_PARAMS.get(
                        key, DEFAULT_GENERIC_TENSOR_PARAMS
                    ),
                )
                group.add_item(tensor)

            dataset_structure.add_group(group)

        return dataset_structure

    def _parse_images_tensor(
        self, sample_compression: str
    ):
        img_config = DEFAULT_IMAGE_TENSOR_PARAMS.copy()

        if self.image_settings.get("link", False):
            img_config["htype"] = "link[image]"

        img_config["sample_compression"] = self.image_settings.get(
            "sample_compression", sample_compression
        )
        name = self.image_settings.get("name", "images")

        return TensorStructure(name=name, primary=True, params=img_config)

    def structure(self, ds: Dataset, use_progress_bar: bool = True):
        (
            img_files,
            _,
            _,
            most_common_compression,
        ) = self.images.parse_images()

        if "sample_compression" not in self.image_settings.keys():
            self.image_settings["sample_compression"] = most_common_compression

        parsed = self._parse_annotation_tensors()
        images_tensor = self._parse_images_tensor(
            sample_compression=most_common_compression
        )

        parsed.add_first_level_tensor(images_tensor)
        parsed.create_structure(ds)

        with ds:
            for ann_file in self.annotation_files:
                coco_file = CocoAnnotation(ann_file)
                id_2_label_mapping = coco_file.id_to_label_mapping
                image_name_to_id = coco_file.image_name_to_id_mapping

                for img_file in tqdm(img_files, disable=not use_progress_bar):
                    try:
                        img_id = image_name_to_id[img_file]
                    except KeyError:
                        logger.warn(
                            f"Could not find the id of image {img_file}, skipping."
                        )
                        continue

                    matching_anns = coco_file.get_annotations_for_image(img_id)
                    group = parsed[
                        self.file_to_group.get(Path(ann_file).stem, Path(ann_file).stem)
                    ]

                    # Get the object to which data will be appended. We need to know if it's first-level tensor, or a group
                    if self.ignore_one_group and len(parsed.structure) == 1:
                        append_obj = ds
                    else:
                        append_obj = ds[group.name]

                    tensors = group.tensors
                    values = {t.name: [] for t in tensors}

                    # Create a list of lists with all the data
                    for ann in matching_anns:
                        for i, tensor in enumerate(tensors):
                            coco_key = self.tensor_to_key.get(tensor.name, tensor.name)
                            value = coco_2_deeplake(
                                coco_key,
                                ann[coco_key],
                                append_obj[tensor.name].meta,
                                category_lookup=id_2_label_mapping,
                            )

                            values[tensor.name].append(value)

                    append_obj.append(values)

                    ds[images_tensor.name].append(
                        deeplake.read(self.images.get_full_path(img_file))
                    )
