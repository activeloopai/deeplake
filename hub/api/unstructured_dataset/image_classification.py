import numpy as np
from pathlib import Path
import os
import glob
from typing import Dict, List, Sequence, Tuple, Union

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

    def get_set_names(self) -> Tuple[str]:
        # TODO: move outside class
        set_names = set()
        for file_path in self._abs_file_paths:
            set_names.add(_set_name_from_path(file_path))
        set_names = sorted(set_names)  # TODO: lexicographical sorting
        return tuple(set_names)

    def get_class_names(self) -> Tuple[str]:
        # TODO: move outside class
        class_names = set()
        for file_path in self._abs_file_paths:
            class_names.add(_class_name_from_path(file_path))
        class_names = sorted(class_names)  # TODO: lexicographical sorting
        return tuple(class_names)

    def structure(self, ds, use_progress_bar: bool = True):
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

            ds.create_tensor(images_tensor_name, htype="image")
            ds.create_tensor(
                labels_tensor_name, htype="class_label", class_names=self.class_names
            )
            # TODO: extra_meta arg should be replaced with `class_names=self.class_names` when htypes are supported

        paths = self._abs_file_paths
        iterator = tqdm(
            paths,
            desc='Ingesting "%s"' % self.source,
            total=len(paths),
            disable=not use_progress_bar,
        )
        for file_path in iterator:
            image = hub.load(file_path, symbolic=False)
            class_name = _class_name_from_path(file_path)
            label = np.array(
                [self.class_names.index(class_name)]
            )  # TODO: should be able to pass just an integer to `tensor.append`

            set_name = _set_name_from_path(file_path) if use_set_prefix else ""
            ds[images_tensor_map[set_name]].append(image)
            ds[labels_tensor_map[set_name]].append(label)

        ds.flush()
        return ds
