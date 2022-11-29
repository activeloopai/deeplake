import os

import json
import numpy as np

from typing import Tuple, List
from pathlib import Path
from collections import defaultdict

import deeplake
from deeplake.htype import HTYPE_SUPPORTED_COMPRESSIONS
from deeplake.util.exceptions import IngestionError
from deeplake.client.log import logger
from deeplake.util.storage import storage_provider_from_path
from deeplake.util.path import convert_pathlib_to_string_if_needed
from deeplake.core.tensor import Tensor


def coco_2_deeplake(coco_key, value, destination_tensor: Tensor, category_lookup=None):
    """Takes a key-value pair from coco data and converts it to data in Deep Lake format
    as per the key types in coco and array shape rules in Deep Lake"""
    dtype = destination_tensor.meta.dtype

    if isinstance(value, list) and len(value) == 0:
        raise Exception("Empty value for key: " + coco_key)

    if coco_key == "bbox":
        assert len(value) == 4
        return np.array(value, dtype=dtype)
    elif coco_key == "segmentation":
        # Make sure there aren't multiple segementations per single value, because multiple things will break
        # # if len(value) > 1:
        #     print("MULTIPLE SEGMENTATIONS PER OBJECT")

        try:
            return np.array(value[0], dtype=dtype).reshape((len(value[0]) // 2), 2)
        except KeyError:
            return np.array([[0, 0]], dtype=dtype)

    elif coco_key == "category_id":
        if category_lookup is None:
            return value
        else:
            return category_lookup[str(value)]

    elif coco_key == "keypoints":
        return np.array(value, dtype=dtype)

    return value


class CocoAnnotation:
    COCO_ANNOTATIONS_KEY = "annotations"
    COCO_IMAGES_KEY = "images"
    COCO_CATEGORIES_KEY = "categories"
    COCO_LICENSES_KEY = "licenses"
    COCO_INFO_KEY = "info"

    COCO_REQUIRED_KEYS = [COCO_CATEGORIES_KEY, COCO_IMAGES_KEY, COCO_ANNOTATIONS_KEY]

    def __init__(self, file_path: str, creds) -> None:
        self.file_path = file_path
        self.root = Path(file_path).parent
        self.file = Path(file_path).name

        root = convert_pathlib_to_string_if_needed(self.root)
        root = root.replace("s3:/", "s3://")

        self.provider = storage_provider_from_path(root, creds=creds)

        self.data = self._load_annotation_data()

    def _load_annotation_data(self):
        """Validates and loads the COCO annotation file."""
        data = json.loads(self.provider.get_bytes(self.file))

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
    def __init__(self, images_directory: str, creds) -> None:
        self.root = images_directory
        self.provider = storage_provider_from_path(
            convert_pathlib_to_string_if_needed(self.root), creds=creds
        )

        (
            self.supported_images,
            self.invalid_files,
            self.extensions,
            self.most_frequent_extension,
        ) = self.parse_images()

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

        for file in self.provider:
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

        most_frequent_extension = (
            max(extensions, key=extensions.get) if len(extensions) > 0 else None
        )

        return (
            supported_images,
            invalid_files,
            list(extensions.keys()),
            most_frequent_extension,
        )

    def get_full_path(self, image_name: str) -> str:
        return os.path.join(self.root, image_name)

    def get_image(self, image: str, destination_tensor: Tensor, creds_key: str):
        if destination_tensor.is_link:
            return deeplake.link(os.path.join(self.root, image), creds_key=creds_key)

        return deeplake.read(os.path.join(self.root, image), storage=self.provider)
