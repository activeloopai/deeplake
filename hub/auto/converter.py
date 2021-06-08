import numpy as np
from pathlib import Path
import os
import glob
from typing import Dict, List, Sequence, Tuple

from tqdm import tqdm

from hub.api.dataset import Dataset
from hub.auto.load import load
from hub.util.path import find_root


IMAGES_TENSOR_NAME = "images"
LABELS_TENSOR_NAME = "labels"
LABEL_NAMES_META_KEY = "class_names"


def _get_file_paths(directory: Path) -> Sequence[str]:
    # TODO: make sure directory is actually a directory

    g = glob.glob(os.path.join(directory, "**"), recursive=True)
    files = [path for path in g if os.path.isfile(path)]
    return files


def _class_name_from_path(path: str) -> str:
    return path.split("/")[-2]


# TODO: rename this
class Converter:
    def __init__(self, unstructured_path: str):
        self.root = Path(find_root(unstructured_path))
        self._file_paths = _get_file_paths(self.root)


    def get_class_names(self) -> Tuple[str]:
        class_names = set()
        for file_path in self._file_paths:
            class_names.add(_class_name_from_path(file_path))
        class_names = sorted(class_names) # TODO: lexicographical sorting
        return tuple(class_names)


    def from_image_classification(self, ds: Dataset, use_tqdm: bool=False):
        class_names = self.get_class_names()

        ds.create_tensor(IMAGES_TENSOR_NAME, htype="image")
        ds.create_tensor(LABELS_TENSOR_NAME, htype="class_label", extra_meta={LABEL_NAMES_META_KEY: class_names})

        iterator = tqdm(self._file_paths, desc="Ingesting image classification dataset", total=len(self._file_paths), disable=not use_tqdm)
        for file_path in iterator:
            image = load(file_path, symbolic=False)
            class_name = _class_name_from_path(file_path)
            label = np.array([class_names.index(class_name)])  # TODO: should be able to pass just an integer to `tensor.append`

            ds[IMAGES_TENSOR_NAME].append(image)
            ds[LABELS_TENSOR_NAME].append(label)

        ds.flush()
        return ds

