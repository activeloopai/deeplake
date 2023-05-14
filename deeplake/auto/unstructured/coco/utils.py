import os

import json
import pathlib

from collections import defaultdict
from typing import Tuple, List, Union, Optional, DefaultDict

import deeplake
from deeplake.htype import HTYPE_SUPPORTED_COMPRESSIONS
from deeplake.util.exceptions import IngestionError
from deeplake.client.log import logger
from deeplake.util.storage import storage_provider_from_path
from deeplake.util.path import convert_pathlib_to_string_if_needed


class CocoAnnotation:
    COCO_ANNOTATIONS_KEY = "annotations"
    COCO_IMAGES_KEY = "images"
    COCO_CATEGORIES_KEY = "categories"
    COCO_LICENSES_KEY = "licenses"
    COCO_INFO_KEY = "info"

    COCO_REQUIRED_KEYS = [COCO_CATEGORIES_KEY, COCO_IMAGES_KEY, COCO_ANNOTATIONS_KEY]

    COCO_SAMPLE_KEYS = set(
        ("id", "image_id", "category_id", "segmentation", "area", "bbox", "iscrowd")
    )

    def __init__(self, file_path: Union[str, pathlib.Path], creds) -> None:
        self.file_path = file_path
        self.root = convert_pathlib_to_string_if_needed(file_path)
        self.file = os.path.basename(self.root)
        self.root = os.path.dirname(self.root)

        self.provider = storage_provider_from_path(self.root, creds=creds)

        self.data = self._load_annotation_data()
        self.id_to_label_mapping = self._get_id_to_label_mapping()
        self.image_name_to_id_mapping = self._get_image_name_to_id_mapping()

    def _load_annotation_data(self):
        """Validates and loads the COCO annotation file."""
        try:
            data = json.loads(self.provider.get_bytes(self.file))
        except KeyError:
            raise IngestionError(
                f"Could not find a JSON annotation file at {self.file_path}."
            )

        for key in self.COCO_REQUIRED_KEYS:
            if key not in data:
                raise IngestionError(
                    f"Invalid annotation file provided. The required key {key} was not found in {self.file_path}."
                )

        return data

    def _get_id_to_label_mapping(self):
        return {str(i["id"]): i["name"] for i in self.categories}

    def _get_image_name_to_id_mapping(self):
        return {i["file_name"]: i["id"] for i in self.images}

    @property
    def file_name(self):
        return pathlib.Path(self.file_path).stem

    @property
    def categories(self):
        return self.data[self.COCO_CATEGORIES_KEY]

    @property
    def annotations(self):
        return self.data[self.COCO_ANNOTATIONS_KEY]

    @property
    def images(self):
        return self.data[self.COCO_IMAGES_KEY]

    def get_annotations_for_image(self, image: str):
        try:
            image_id = self.image_name_to_id_mapping[image]
        except KeyError:
            raise IngestionError(
                f"Could not find corresponding image_id for {image} in {self.file_name} file."
            )
        return list(
            filter(
                lambda item: item["image_id"] == image_id,
                self.annotations,
            )
        )


class CocoImages:
    def __init__(self, images_directory: Union[str, pathlib.Path], creds) -> None:
        self.root = convert_pathlib_to_string_if_needed(images_directory)
        self.provider = storage_provider_from_path(self.root, creds=creds)

        (
            self.supported_images,
            self.most_frequent_extension,
        ) = self.parse_images()

    def parse_images(self) -> Tuple[List[str], Optional[str]]:
        """Parses the given directory to generate a list of image paths.
        Returns:
            A tuple with, respectively, list of supported images, the most frequent extension
        """
        supported_image_extensions = tuple(
            "." + fmt for fmt in HTYPE_SUPPORTED_COMPRESSIONS["image"] + ["jpg"]
        )
        supported_images = []
        invalid_files = []
        extensions: DefaultDict[str, int] = defaultdict(int)

        for file in self.provider:
            if file.lower().endswith(supported_image_extensions):
                supported_images.append(file)
                ext = file.rsplit(".", 1)[1]
                extensions[ext] += 1
            else:
                invalid_files.append(file)

        if len(invalid_files) > 0:
            logger.warning(
                f"Encountered {len(invalid_files)} unsupported files in images directory."
                + "\nUp to first 10 invalid files are:\n"
                + "\n".join(invalid_files[0:10])
            )

        if len(supported_images) == 0:
            raise IngestionError(
                f"No supported images found in {self.root}. Supported extensions are: {supported_image_extensions}"
            )

        most_frequent_extension = max(
            extensions, key=lambda k: extensions[k], default=None
        )

        return (
            supported_images,
            most_frequent_extension,
        )

    def get_full_path(self, image_name: str) -> str:
        return os.path.join(self.root, image_name)

    def get_image(self, image: str, linked: bool, creds_key: Optional[str] = None):
        if linked:
            return deeplake.link(self.get_full_path(image), creds_key=creds_key)

        return deeplake.read(self.get_full_path(image), storage=self.provider)
