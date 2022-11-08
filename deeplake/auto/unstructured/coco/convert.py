import os

import json
import numpy as np

from typing import Tuple, List
from pathlib import Path
from collections import defaultdict

from deeplake.htype import HTYPE_SUPPORTED_COMPRESSIONS
from deeplake.util.exceptions import IngestionError
from deeplake.client.log import logger


def coco_2_deeplake(coco_key, value, tensor_meta, category_lookup=None):
    """Takes a key-value pair from coco data and converts is to data in Deep Lake format
    as per the key types in coco and array shape rules in Deep Lake"""
    dtype = tensor_meta.dtype

    if coco_key == "bbox":
        return np.array(value).astype(dtype)
    elif coco_key == "segmentation":
        if len(value) == 0:
            return np.array([]).astype(dtype)

        # Make sure there aren't multiple segementations per single value, because multiple things will break
        if len(value) > 1:
            print("MULTIPLE SEGMENTATIONS PER OBJECT")

        try:
            return np.array(value[0]).reshape((len(value[0]) // 2), 2).astype(dtype)
        except KeyError:
            print("KEY ERROR", value)
            return np.array([]).astype(dtype)

    elif coco_key == "category_id":
        if category_lookup is None:
            return value
        else:
            return category_lookup[str(value)]

    elif coco_key == "keypoints":
        return np.array(value).astype(dtype)


class CocoAnnotation:
    COCO_ANNOTATIONS_KEY = "annotations"
    COCO_IMAGES_KEY = "images"
    COCO_CATEGORIES_KEY = "categories"
    COCO_LICENSES_KEY = "licenses"
    COCO_INFO_KEY = "info"

    COCO_REQUIRED_KEYS = [COCO_CATEGORIES_KEY, COCO_IMAGES_KEY, COCO_ANNOTATIONS_KEY]

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.data = self._load_annotation_data()

    def _load_annotation_data(self):
        """Validates and loads the COCO annotation file."""
        with open(self.file_path, "r") as f:
            data = json.load(f)

        for key in self.COCO_REQUIRED_KEYS:
            if key not in data:
                raise IngestionError(
                    f"Invalid annotation file provided. The required key {key} was not found in {self.file_path}."
                )

        return data

    @property
    def categories(self):
        return self.data[self.COCO_CATEGORIES_KEY]

    @property
    def annotations(self):
        return self.data[self.COCO_ANNOTATIONS_KEY]

    @property
    def images(self):
        return self.data[self.COCO_IMAGES_KEY]

    @property
    def id_to_label_mapping(self):
        return {str(i["id"]): i["name"] for i in self.categories}

    @property
    def image_name_to_id_mapping(self):
        return {i["file_name"]: i["id"] for i in self.images}

    def get_annotations_for_image(self, image_id: str):
        return list(
            filter(
                lambda item: item["image_id"] == image_id,
                self.annotations,
            )
        )


class CocoImages:
    def __init__(self, images_directory: str) -> None:
        self.root = images_directory

    def parse_images(self) -> Tuple[List[str], List[str], List[str], str]:
        """Parses the given directory to generate a list of image paths.
        Returns:
            A tuple with, respectively, list of supported images, list of encountered invalid files, list of encountered extensions and the most frequent extension
        """
        supported_image_extensions = tuple(
            HTYPE_SUPPORTED_COMPRESSIONS["image"] + ["jpg"]
        )
        supported_images = []
        invalid_files = []
        extensions = defaultdict(int)
        for file in os.listdir(self.root):
            if file.endswith(supported_image_extensions):
                supported_images.append(file)
                ext = Path(file).suffix[1:]  # Get extension without the . symbol
                extensions[ext] += 1
            else:
                invalid_files.append(file)

        if len(invalid_files) > 0:
            logger.warn(
                f"Encountered {len(invalid_files)} unsupported files in images directory."
            )

        return (
            supported_images,
            invalid_files,
            list(extensions.keys()),
            max(extensions, key=lambda k: extensions[k]),
        )
