import deeplake

from pathlib import Path
from typing import List, Union, Dict, Optional
from itertools import chain
from collections import OrderedDict

from deeplake.core.dataset import Dataset
from deeplake.util.exceptions import IngestionError

from ..base import UnstructuredDataset
from ..util import DatasetStructure, GroupStructure, TensorStructure
from .utils import CocoAnnotation, CocoImages
from .convert import coco_to_deeplake

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
        self.ignore_keys = set(ignore_keys or [])

        self.key_to_tensor = key_to_tensor_mapping or {}
        self._validate_key_mapping()
        self.tensor_to_key = {v: k for k, v in self.key_to_tensor.items()}
        # If a key is not mapped to a tensor, map it to itself
        self.tensor_to_key.update(
            {
                k: k
                for k in CocoAnnotation.COCO_SAMPLE_KEYS
                - set(self.key_to_tensor.keys())
                - self.ignore_keys
            }
        )

        self.file_to_group = file_to_group_mapping or {}
        self.file_to_group = {Path(k).stem: v for k, v in self.file_to_group.items()}
        self._validate_group_mapping()

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

    def _add_annotation_tensors(
        self,
        structure: DatasetStructure,
        inspect_limit: int = 1000000,
    ):
        if inspect_limit < 1:
            inspect_limit = 1

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

    def _add_images_tensor(self, structure: DatasetStructure):
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

    def _ingest_images(
        self,
        ds: Dataset,
        images: List[str],
        progressbar: bool = True,
        num_workers: int = 0,
    ):
        images_tensor_name = self.image_settings.get("name")
        images_tensor = ds[images_tensor_name]
        samples = OrderedDict((image, None) for image in images)
        creds_key = self.image_settings.get("creds_key", None)

        if creds_key is not None:
            ds.add_creds_key(creds_key, managed=True)

        @deeplake.compute
        def append_images(image, _, samples):
            samples[image] = self.images.get_image(
                image,
                destination_tensor=images_tensor,
                creds_key=creds_key,
            )

        append_images(samples).eval(
            images, ds, progressbar=progressbar, num_workers=num_workers
        )
        images_tensor.extend(list(samples.values()), progressbar=progressbar)

    def _get_sample(
        self,
        image_file: str,
        ds: Dataset,
        coco_file: CocoAnnotation,
    ):
        id_to_label = coco_file.id_to_label_mapping
        image_name_to_id = coco_file.image_name_to_id_mapping
        image_id = image_name_to_id[image_file]
        matching_annotations = coco_file.get_annotations_for_image(image_id)
        group_tensors = [
            t for t in ds.tensors if t in self.tensor_to_key and not ds[t].meta.hidden
        ]
        sample: Dict[str, List] = {k: [] for k in group_tensors}

        for annotation in matching_annotations:
            for tensor_name in group_tensors:
                coco_key = self.tensor_to_key.get(tensor_name, tensor_name)
                value = coco_to_deeplake(
                    coco_key,
                    annotation[coco_key],
                    ds[tensor_name],
                    category_lookup=id_to_label,
                )

                sample[tensor_name].append(value)

        return sample

    def prepare_structure(self, inspect_limit: int = 1000000) -> DatasetStructure:
        structure = DatasetStructure(ignore_one_group=self.ignore_one_group)
        self._add_annotation_tensors(structure, inspect_limit=inspect_limit)
        self._add_images_tensor(structure)

        return structure

    def structure(self, ds: Dataset, progressbar: bool = True, num_workers: int = 0):  # type: ignore
        image_files = self.images.supported_images

        with ds:
            self._ingest_images(ds, image_files, progressbar, num_workers)

            for ann_file in self.annotation_files:
                coco_file = CocoAnnotation(ann_file, creds=self.creds)

                group_prefix = self.file_to_group.get(
                    Path(ann_file).stem, Path(ann_file).stem
                )
                append_destination = ds

                # Get the object to which data will be appended. We need to know if it's first-level tensor, or a group
                if self.ignore_one_group and len(ds.groups) == 1:
                    group_prefix = ""

                if group_prefix:
                    append_destination = ds[group_prefix]

                @deeplake.compute
                def process_annotations(image, _, values):
                    values[image] = self._get_sample(
                        image,
                        append_destination,
                        coco_file,
                    )

                values: Dict[str, Dict] = OrderedDict((image, None) for image in image_files)  # type: ignore

                process_annotations(values).eval(
                    image_files,
                    append_destination,
                    num_workers=num_workers,
                    progressbar=progressbar,
                )
                samples = {
                    key: [item[key] for item in values.values()]
                    for key in values[image_files[0]].keys()
                }

                append_destination.extend(samples)
