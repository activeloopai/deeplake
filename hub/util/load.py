import re
import numpy as np
import pathlib
import exiftool  # type: ignore
from typing import Callable, Dict, List, Union
from hub.util.exceptions import (
    HubAutoUnsupportedFileExtensionError,
    SampleCorruptedError,
    ImageReadError,
)
from PIL import Image  # type: ignore


IMAGE_SUFFIXES: List[str] = [".jpeg", ".jpg", ".png"]
SUPPORTED_SUFFIXES: List[str] = IMAGE_SUFFIXES


class SymbolicSample:
    def __init__(self, path: str):
        self.path = pathlib.Path(path)
        self._array = None

    @property
    def name(self):
        return self.path.split("/")[-1]

    @property
    def shape(self):
        # TODO: raise good exception if `numpy` wasn't called
        return self._array.shape

    @property
    def dtype(self):
        # TODO: raise good exception if `numpy` wasn't called
        return self._array.dtype.name

    @property
    def compression(self) -> str:
        # TODO: raise exception if `numpy` wasn't called
        return self._compression

    def raw_bytes(self):
        with open(self.path, "rb") as image_file:
            return image_file.read()

    def numpy(self) -> np.ndarray:
        # TODO: explain this
        if self._array is not None:
            return self._array

        suffix = self.path.suffix.lower()

        if suffix in IMAGE_SUFFIXES:
            img = Image.open(self.path)

            # TODO: mention in docstring that if this loads correctly, the meta is assumed to be valid
            try:
                self._array = np.array(img)
            except:
                # TODO: elaborate on why it corrupted
                raise SampleCorruptedError(self.path)

            # TODO: set meta values
            self._compression = img.format
            return self._array

        raise HubAutoUnsupportedFileExtensionError(self._suffix, SUPPORTED_SUFFIXES)

    # TODO: __str__


def load(path: Union[str, pathlib.Path]) -> SymbolicSample:
    # TODO: mention that you can do `.numpy()` on this output to make it extremely easy to use
    return SymbolicSample(path)
