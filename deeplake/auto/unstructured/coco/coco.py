import os
import deeplake

from pathlib import Path
from typing import List, Union, Dict, Optional
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
        key_to_tensor_mapping: Optional[Dict] = None,
        file_to_group_mapping: Optional[Dict] = None,
        ignore_one_group: bool = False,
        ignore_keys: Optional[List[str]] = None,
        image_settings: Optional[Dict] = None,
        creds: Optional[Dict] = None,
    ):
        super().__init__(source)
        self.creds = creds
        self.images = CocoImages(images_directory=source, creds=creds)
        self.linked_images = False

        self.annotation_files = (
            [annotation_files]
            if not isinstance(annotation_files, list)
            else annotation_files
        )
        self.ignore_one_group = ignore_one_group

        self.key_to_tensor = key_to_tensor_mapping or {}
        self._validate_key_mapping()
        self.tensor_to_key = {v: k for k, v in self.key_to_tensor.items()}

        self.file_to_group = {Path(k).stem: v for k, v in file_to_group_mapping.items()}
        self._validate_group_mapping()

        self.ignore_keys = ignore_keys or []
        self.image_settings = (
            image_settings if image_settings is not None else {"name": "images"}
        )
        self._validate_image_settings()

    def _validate_key_mapping(self):
        if len(self.key_to_tensor.values()) != len(set(self.key_to_tensor.values())):
            raise IngestionError("Keys must be mapped to unique tensor names.")

    def _validate_group_mapping(self):
        if len(self.file_to_group.values()) != len(set(self.file_to_group.values())):
            raise IngestionError("File names must be mapped to unique group names.")

    def _validate_image_settings(self):
        if "name" not in self.image_settings:
            raise IngestionError(
                "Image settings must contain a name for the image tensor."
            )

    def _parse_annotation_tensors(
        self,
        structure: DatasetStructure,
        inspect_limit: int = 1000000,
    ):
        for ann_file in self.annotation_files:
            coco_file = CocoAnnotation(file_path=ann_file, creds=self.creds)
            annotations = coco_file.annotations
            file_name = Path(ann_file).stem
            keys_in_group = set(chain.from_iterable(annotations[:inspect_limit]))

            group_name = self.file_to_group.get(file_name, file_name)
            group = GroupStructure(group_name)

            for key in keys_in_group:
                if key in self.ignore_keys:
                    continue
                tensor_name = self.key_to_tensor.get(key, key)

                tensor = TensorStructure(
                    name=tensor_name,
                    params=DEFAULT_COCO_TENSOR_PARAMS.get(
                        key, DEFAULT_GENERIC_TENSOR_PARAMS
                    ),
                )
                group.add_item(tensor)

            structure.add_group(group)

    def _parse_images_tensor(self, structure: DatasetStructure):
        img_config = DEFAULT_IMAGE_TENSOR_PARAMS.copy()

        if self.image_settings.get("linked", False):
            img_config["htype"] = "link[image]"

        img_config["sample_compression"] = self.image_settings.get(
            "sample_compression", self.images.most_frequent_extension
        )
        name = self.image_settings.get("name", "images")

        structure.add_first_level_tensor(
            TensorStructure(name=name, primary=True, params=img_config)
        )

    def prepare_structure(self, inspect_limit: int = 1000000) -> DatasetStructure:
        structure = DatasetStructure(ignore_one_group=self.ignore_one_group)
        self._parse_annotation_tensors(structure, inspect_limit=inspect_limit)
        self._parse_images_tensor(structure)

        return structure

    def process_single_image(
        self,
        img_file,
        ds: Dataset,
        image_name_to_id,
        id_2_label,
        coco_file,
        group_prefix,
    ):
        img_id = image_name_to_id[img_file]
        matching_anns = coco_file.get_annotations_for_image(img_id)
        group_tensors = ds[group_prefix].tensors if group_prefix else ds.tensors
        values = {t.key: [] for t in group_tensors.values()}
        image_tensor = self.image_settings.get("name")

        values[image_tensor] = self.images.get_image(
            img_file,
            destination_tensor=ds[image_tensor],
            creds_key=self.image_settings.get("creds_key"),
        )

        for ann in matching_anns:
            for tensor_name, tensor in group_tensors.items():
                coco_key = self.tensor_to_key.get(tensor_name, tensor_name)
                value = coco_2_deeplake(
                    coco_key,
                    ann[coco_key],
                    ds[tensor.key],
                    category_lookup=id_2_label,
                )

                values[tensor.key].append(value)

        return values

    def structure(self, ds: Dataset, progressbar: bool = True, num_workers: int = 0):
        img_files = self.images.supported_images

        with ds:
            for ann_file in self.annotation_files:
                coco_file = CocoAnnotation(ann_file, creds=self.creds)
                id_2_label_mapping = coco_file.id_to_label_mapping
                image_name_to_id = coco_file.image_name_to_id_mapping

                group_prefix = self.file_to_group.get(
                    Path(ann_file).stem, Path(ann_file).stem
                )

                # Get the object to which data will be appended. We need to know if it's first-level tensor, or a group
                if self.ignore_one_group and len(ds.groups) == 1:
                    group_prefix = ""

                @deeplake.compute
                def process_annotations(image, _, values):
                    values.append(
                        self.process_single_image(
                            image,
                            ds,
                            image_name_to_id,
                            id_2_label_mapping,
                            coco_file,
                            group_prefix,
                        )
                    )

                values = []
                process_annotations(values).eval(
                    img_files, ds, num_workers=num_workers, progressbar=progressbar
                )

                @deeplake.compute
                def append_annotations(s, ds):
                    ds.append(s)

                append_annotations().eval(
                    values, ds, num_workers=num_workers, progressbar=progressbar
                )
