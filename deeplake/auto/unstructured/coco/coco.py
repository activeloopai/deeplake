import deeplake

from pathlib import Path
from typing import List, Union, Dict, Optional
from itertools import chain

from deeplake.core.dataset import Dataset
from deeplake.core.tensor import Tensor
from deeplake.util.exceptions import IngestionError

from ..base import UnstructuredDataset
from ..util import DatasetStructure, GroupStructure, TensorStructure
from .utils import CocoAnnotation, CocoImages
from .convert import coco_to_deeplake

from random import shuffle as rshuffle

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
        image_params: Optional[Dict] = None,
        image_creds_key: Optional[str] = None,
        creds: Optional[Union[str, Dict]] = None,
    ):
        super().__init__(source)
        self._creds = creds
        self._image_creds_key = image_creds_key
        self.image_params = image_params or {}
        self.images = CocoImages(images_directory=source, creds=creds)

        if not isinstance(annotation_files, list):
            annotation_files = [annotation_files]
        self.annotation_files = [
            CocoAnnotation(file, self._creds) for file in annotation_files
        ]

        self.ignore_one_group = ignore_one_group
        self.preserve_flat_structure = (
            self.ignore_one_group and len(annotation_files) == 1
        )

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

    def _validate_key_mapping(self):
        if len(self.key_to_tensor.values()) != len(set(self.key_to_tensor.values())):
            raise IngestionError("Keys must be mapped to unique tensor names.")

    def _validate_group_mapping(self):
        if len(self.file_to_group.values()) != len(set(self.file_to_group.values())):
            raise IngestionError("File names must be mapped to unique group names.")

    def _get_full_tensor_name(self, group: str, tensor: str):
        if self.preserve_flat_structure:
            return tensor

        return f"{group}/{tensor}"

    def _add_annotation_tensors(
        self,
        structure: DatasetStructure,
        inspect_limit: int = 1000000,
    ):
        if inspect_limit < 1:
            inspect_limit = 1

        for coco_file in self.annotation_files:
            annotations = coco_file.annotations
            file_name = coco_file.file_name
            keys_in_group = (
                set(chain.from_iterable(annotations[:inspect_limit])) - self.ignore_keys
            )

            group_name = self.file_to_group.get(file_name, file_name)
            group = GroupStructure(group_name)

            for key in keys_in_group:
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
        images_tensor_params = {**DEFAULT_IMAGE_TENSOR_PARAMS, **self.image_params}
        name = images_tensor_params.pop("name")

        # If the user has not explicitly specified a compression, try to infer it, or use default one
        if "sample_compression" not in self.image_params:
            images_tensor_params["sample_compression"] = (
                self.images.most_frequent_extension
                or DEFAULT_IMAGE_TENSOR_PARAMS["sample_compression"]
            )

        structure.add_first_level_tensor(
            TensorStructure(name, params=images_tensor_params)
        )
        self._images_tensor_name = name

    def prepare_structure(self, inspect_limit: int = 1000000) -> DatasetStructure:
        structure = DatasetStructure(ignore_one_group=self.ignore_one_group)
        self._add_annotation_tensors(structure, inspect_limit=inspect_limit)
        self._add_images_tensor(structure)

        self._structure = structure
        return structure

    def structure(self, ds: Dataset, progressbar: bool = True, num_workers: int = 0, shuffle: bool = True):  # type: ignore
        image_files = self.images.supported_images

        if shuffle:
            rshuffle(image_files)

        tensors = ds.tensors

        if self._image_creds_key is not None:
            ds.add_creds_key(self._image_creds_key, managed=True)

        @deeplake.compute
        def append_samples(
            image: str,
            ds: Dataset,
            tensors: Dict[str, Tensor],
        ):
            images_tensor_name = self._images_tensor_name
            full_sample: Dict[str, List] = {key: [] for key in self._structure.all_keys}
            full_sample[images_tensor_name] = self.images.get_image(
                image,
                tensors[images_tensor_name].is_link,
                creds_key=self._image_creds_key,
            )

            for coco_file in self.annotation_files:
                file_name = coco_file.file_name
                id_to_label = coco_file.id_to_label_mapping
                matching_annotations = coco_file.get_annotations_for_image(image)
                group_prefix = self.file_to_group.get(file_name, file_name)

                for annotation in matching_annotations:
                    for tensor_name in self.tensor_to_key:
                        full_name = self._get_full_tensor_name(
                            group_prefix, tensor_name
                        )

                        coco_key = self.tensor_to_key.get(tensor_name, tensor_name)
                        value = coco_to_deeplake(
                            coco_key,
                            annotation[coco_key],
                            tensors[full_name],
                            category_lookup=id_to_label,
                        )
                        full_sample[full_name].append(value)

            ds.append(full_sample)

        append_samples(tensors).eval(
            image_files, ds, num_workers=num_workers, progressbar=progressbar
        )
