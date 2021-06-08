from hub.util.path import find_root
from pathlib import Path
import os
import glob
from typing import Dict, List, Sequence

from hub.api.dataset import Dataset


IMAGES_TENSOR_NAME = "images"



def _get_file_paths(directory: Path) -> Sequence[str]:
    # TODO: make sure directory is actually a directory

    g = glob.glob(os.path.join(directory, "**"), recursive=True)
    files = [path for path in g if os.path.isfile(path)]
    return files


# TODO: rename this
class Converter:
    def __init__(self, unstructured_path: str):
        self.root = Path(find_root(unstructured_path))

    # TODO: rename this
    def write_to(self, ds: Dataset):
        files = _get_file_paths(self.root)
        print(files[:5])