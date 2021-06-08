import numpy as np
import pathlib
from typing import Callable, Dict, List, Union

from PIL import Image


IMAGE_SUFFIXES: List[str] = [".jpeg", ".jpg", ".png"]


class SymbolicSample:
    def __init__(self, path: str, loaders: Dict[str, Callable]):
        self.loaders = loaders
        self.path = path

    def load(self) -> Dict[str, np.ndarray]:
        sample = {}
        for tensor_name, loader in self.loaders.items():
            sample[tensor_name] = loader(self.path)
        return sample


def _load_image(image_path: str) -> np.ndarray:
    img = Image.open(image_path)
    return np.array(img)


def _class_name_from_path(path: str) -> str:
    return "DUMMY"


def load(path: Union[str, pathlib.Path], symbolic=False) -> Union[Callable, np.ndarray]:
    path = pathlib.Path(path)
    
    suffixes = path.suffixes

    if len(suffixes) != 1:
        raise Exception() # TODO: handle != 1

    suffix = suffixes[0].lower()

    if suffix in IMAGE_SUFFIXES:
        if symbolic:
            return SymbolicSample(path, {"image": _load_image, "class_name": _class_name_from_path})
        return _load_image(path)
        
    raise Exception()  # TODO: exceptions.py