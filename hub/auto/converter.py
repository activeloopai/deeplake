from hub.api.dataset import Dataset
import os
import glob
from typing import Dict, List, Sequence


IMAGES_TENSOR_NAME = "images"




def _find_root(path: str):
    # TODO
    return path


def _get_file_paths(directory: str) -> Sequence[str]:
    # TODO: make sure directory is actually a directory

    g = glob.glob(os.path.join(directory, "**"), recursive=True)
    files = [path for path in g if os.path.isfile(path)]
    return files


def _tensor_name_from_file_path(file_path: str):
    # TODO:
    if file_path.endswith(".jpg"):
        return IMAGES_TENSOR_NAME

    raise NotImplementedError()  # TODO: exceptions.py


def _extract_tensors_to_files_dict(file_paths: Sequence[str]):
    tensors_to_files: Dict[list] = {}

    for file_path in file_paths:
        tensor_name = _tensor_name_from_file_path(file_path)
        if tensor_name not in tensors_to_files:
            tensors_to_files[tensor_name] = []
        tensors_to_files[tensor_name].append(file_path)

    return tensors_to_files


# TODO: rename this
class Converter:
    def __init__(self, unstructured_path: str):
        # TODO: find dataset root (lowest common denominator of files)

        self.root = _find_root(unstructured_path)

        
    # TODO: rename this
    def write_to(self, ds: Dataset):
        files = _get_file_paths(self.root)
        tensors_to_files = _extract_tensors_to_files_dict(files)
        print(tensors_to_files.keys())