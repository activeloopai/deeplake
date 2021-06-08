from hub.util.path import find_root
from pathlib import Path
import os
import glob
from typing import Dict, List, Sequence, Tuple

from hub.api.dataset import Dataset
from hub.auto.load import load


IMAGES_TENSOR_NAME = "images"


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


    def from_image_classification(self, ds: Dataset):
        class_names = self.get_class_names()

        for file_path in self._file_paths:
            image = load(file_path, symbolic=False)
            class_name = _class_name_from_path(file_path)
            label = class_names.index(class_name)
            print(class_name, label, image.shape)

