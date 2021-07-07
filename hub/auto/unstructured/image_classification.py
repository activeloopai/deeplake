import warnings
import numpy as np
from pathlib import Path
import os
import glob
from typing import Dict, List, Sequence, Tuple, Union

from hub.util.exceptions import TensorInvalidSampleShapeError
from hub import Dataset

from tqdm import tqdm

from .base import UnstructuredDataset

import hub

IMAGES_TENSOR_NAME = "images"
LABELS_TENSOR_NAME = "labels"


def _get_file_paths(
    directory: Path, relative_to: Union[str, Path] = ""
) -> Sequence[str]:
    # TODO: make sure directory is actually a directory
    g = glob.glob(os.path.join(directory, "**"), recursive=True)
    file_paths = []
    for path_str in g:
        if os.path.isfile(path_str):
            path = Path(path_str)
            if relative_to:
                relative_path = Path(path).relative_to(directory)
            else:
                relative_path = path
            file_paths.append(relative_path)
    return file_paths


def _class_name_from_path(path: Path) -> str:
    return path.parts[-2]


def _set_name_from_path(path: Path) -> str:
    return path.parts[-3]


class ImageClassification(UnstructuredDataset):
    def __init__(self, source: str):
        # TODO: should support any `StorageProvider`. right now only local files can be converted
        super().__init__(source)

        self._abs_file_paths = _get_file_paths(self.source)
        self._rel_file_paths = _get_file_paths(self.source, relative_to=self.source)

        self.set_names = self.get_set_names()
        self.class_names = self.get_class_names()

    # TODO: make lazy/memoized property
    def get_set_names(self) -> Tuple[str]:
        # TODO: move outside class
        set_names = set()
        for file_path in self._abs_file_paths:
            set_names.add(_set_name_from_path(file_path))
        set_names = sorted(set_names)  # TODO: lexicographical sorting
        return tuple(set_names)

    # TODO: make lazy/memoized property
    def get_class_names(self) -> Tuple[str]:
        # TODO: move outside class
        class_names = set()
        for file_path in self._abs_file_paths:
            class_names.add(_class_name_from_path(file_path))
        class_names = sorted(class_names)  # TODO: lexicographical sorting
        return tuple(class_names)

    def structure(
        self, ds: Dataset, use_progress_bar: bool = True, image_tensor_args: dict = {}
    ):
        images_tensor_map = {}
        labels_tensor_map = {}

        use_set_prefix = len(self.set_names) > 1

        for set_name in self.set_names:
            if not use_set_prefix:
                set_name = ""

            images_tensor_name = os.path.join(set_name, IMAGES_TENSOR_NAME)
            labels_tensor_name = os.path.join(set_name, LABELS_TENSOR_NAME)
            images_tensor_map[set_name] = images_tensor_name
            labels_tensor_map[set_name] = labels_tensor_name

            # TODO: infer sample_compression
            ds.create_tensor(images_tensor_name, htype="image", **image_tensor_args)
            ds.create_tensor(
                labels_tensor_name,
                htype="class_label",
                class_names=self.class_names,
            )

        with ds:
            paths = self._abs_file_paths
            iterator = tqdm(
                paths,
                desc='Ingesting "%s"' % self.source,
                total=len(paths),
                disable=not use_progress_bar,
            )
            for file_path in iterator:
                image = hub.load(file_path)
                class_name = _class_name_from_path(file_path)

                label = np.uint32(self.class_names.index(class_name))

                set_name = _set_name_from_path(file_path) if use_set_prefix else ""

                # TODO: try to get all len(shape)s to match.
                # if appending fails because of a shape mismatch, expand dims (might also fail)
                try:
                    ds[images_tensor_map[set_name]].append(image)
                except TensorInvalidSampleShapeError:
                    im = image.array
                    reshaped_image = np.expand_dims(im, -1)
                    ds[images_tensor_map[set_name]].append(reshaped_image)
                except Exception as e:
                    if hasattr(e, "message"):
                        resason = e.message
                    else:
                        reason = "Unknown"

                    warnings.warn(f"[Skipping] Could not upload sample '{file_path}'. Reason: {reason}")
                    continue

                ds[labels_tensor_map[set_name]].append(label)

            return ds
