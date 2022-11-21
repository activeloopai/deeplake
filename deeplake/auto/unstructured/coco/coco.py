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
        key_to_tensor_mapping: Dict = {},
        file_to_group_mapping: Dict = {},
        ignore_one_group: bool = False,
        ignore_keys: Union[str, List[str]] = [],
        image_settings: Dict = {},
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
        ds: Dataset,
        inspect_limit: int = 1000000,
    ) -> DatasetStructure:
        """Return all the tensors and groups that should be created for this dataset"""
        dataset_structure = DatasetStructure(ignore_one_group=self.ignore_one_group)

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

                if group_name + "/" + tensor_name in ds:
                    continue

                tensor = TensorStructure(
                    name=tensor_name,
                    params=DEFAULT_COCO_TENSOR_PARAMS.get(
                        key, DEFAULT_GENERIC_TENSOR_PARAMS
                    ),
                )
                group.add_item(tensor)

            dataset_structure.add_group(group)

        return dataset_structure

    def _parse_images_tensor(self, sample_compression: str):
        img_config = DEFAULT_IMAGE_TENSOR_PARAMS.copy()

        if self.image_settings.get("link", False):
            img_config["htype"] = "link[image]"

        img_config["sample_compression"] = self.image_settings.get(
            "sample_compression", sample_compression
        )
        name = self.image_settings.get("name", "images")

        return TensorStructure(name=name, primary=True, params=img_config)

    def generate_images_data(
        self, images, ann_file, parsed, image, append_obj, images_dir
    ):
        coco_file = CocoAnnotation(ann_file, creds=self.creds)
        id_2_label_mapping = coco_file.id_to_label_mapping
        image_name_to_id = coco_file.image_name_to_id_mapping

        group = parsed[self.file_to_group.get(Path(ann_file).stem, Path(ann_file).stem)]
        group_prefix = group.name
        tensors = group.tensors

        for img_file in images:
            img_id = image_name_to_id[img_file]
            matching_anns = coco_file.get_annotations_for_image(img_id)
            values = {group_prefix + "/" + t.name: [] for t in tensors}
            values[image.name] = deeplake.read(images_dir + "/" + img_file)

            # Create a list of lists with all the data
            for ann in matching_anns:
                for tensor in tensors:
                    coco_key = self.tensor_to_key.get(tensor.name, tensor.name)
                    value = coco_2_deeplake(
                        coco_key,
                        ann[coco_key],
                        append_obj[tensor.name].meta,
                        category_lookup=id_2_label_mapping,
                    )

                    values[group_prefix + "/" + tensor.name].append(value)

            yield values

    def process_single_image(
        self,
        img_file,
        append_obj,
        image_name_to_id,
        id_2_label,
        coco_file,
        tensors,
        group_prefix,
        image,
    ):
        img_id = image_name_to_id[img_file]
        matching_anns = coco_file.get_annotations_for_image(img_id)
        values = {group_prefix + "/" + t.name: [] for t in tensors}
        values[image.name] = deeplake.read(self.images.get_full_path(img_file))

        for ann in matching_anns:
            for tensor in tensors:
                coco_key = self.tensor_to_key.get(tensor.name, tensor.name)
                value = coco_2_deeplake(
                    coco_key,
                    ann[coco_key],
                    append_obj[tensor.name].meta,
                    category_lookup=id_2_label,
                )

                values[group_prefix + "/" + tensor.name].append(value)

        return values

    def structure(self, ds: Dataset, progressbar: bool = True):
        (
            img_files,
            _,
            _,
            most_common_compression,
        ) = self.images.parse_images()

        if "sample_compression" not in self.image_settings.keys():
            self.image_settings["sample_compression"] = most_common_compression

        parsed = self._parse_annotation_tensors(ds)
        images_tensor = self._parse_images_tensor(
            sample_compression=most_common_compression
        )

        parsed.add_first_level_tensor(images_tensor)
        parsed.create_structure(ds)

        with ds:
            for ann_file in self.annotation_files:
                coco_file = CocoAnnotation(ann_file, creds=self.creds)
                id_2_label_mapping = coco_file.id_to_label_mapping
                image_name_to_id = coco_file.image_name_to_id_mapping

                group = parsed[
                    self.file_to_group.get(Path(ann_file).stem, Path(ann_file).stem)
                ]
                tensors = group.tensors

                # Get the object to which data will be appended. We need to know if it's first-level tensor, or a group
                if self.ignore_one_group and len(parsed.structure) == 1:
                    append_obj = ds
                    group_prefix = ""
                else:
                    append_obj = ds[group.name]
                    group_prefix = group.name

                @deeplake.compute
                def process_annotations(image, _, values):
                    values.append(
                        self.process_single_image(
                            image,
                            append_obj,
                            image_name_to_id,
                            id_2_label_mapping,
                            coco_file,
                            tensors,
                            group_prefix,
                            images_tensor,
                        )
                    )

                values = []
                process_annotations(values).eval(img_files, ds)

                @deeplake.compute
                def append_annotations(s, ds):
                    ds.append(s)

                append_annotations().eval(values, ds)
