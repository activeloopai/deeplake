import numpy as np
import pathlib
from typing import Callable, Dict, List, Union
from hub.util.exceptions import HubAutoUnsupportedFileExtensionError

from PIL import Image


IMAGE_SUFFIXES: List[str] = [".jpeg", ".jpg", ".png"]
SUPPORTED_SUFFIXES: List[str] = IMAGE_SUFFIXES


def _load_image(image_path: str) -> np.ndarray:
    img = Image.open(image_path)
    return np.array(img)


def load(path: Union[str, pathlib.Path], symbolic=False) -> Union[Callable, np.ndarray]:
    path = pathlib.Path(path)

    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        if symbolic:
            # TODO: symbolic loading (for large samples)
            raise NotImplementedError("Symbolic `hub.load` not implemented.")
        return _load_image(path)
        
    raise HubAutoUnsupportedFileExtensionError(suffix, SUPPORTED_SUFFIXES)