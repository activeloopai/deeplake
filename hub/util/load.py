from hub.constants import UNCOMPRESSED
from hub.core.chunk_engine.flatten import row_wise_to_bytes
import re
import numpy as np
import pathlib
import exiftool  # type: ignore
from typing import Callable, Dict, List, Optional, Tuple, Union
from hub.util.exceptions import (
    HubAutoUnsupportedFileExtensionError,
    SampleCorruptedError,
    ImageReadError,
)
from PIL import Image  # type: ignore


IMAGE_SUFFIXES: List[str] = [".jpeg", ".jpg", ".png"]
SUPPORTED_SUFFIXES: List[str] = IMAGE_SUFFIXES


class Sample:
    path: Optional[pathlib.Path]

    def __init__(
        self,
        path: str = None,
        array: np.ndarray = None,
        compression: str = UNCOMPRESSED,
    ):
        if (path is None) == (array is None):
            raise ValueError("Must pass either `path` or `array`.")

        if path is not None:
            self.path = pathlib.Path(path)
            self._array = None
            if compression != UNCOMPRESSED:
                # TODO: maybe this should be possible? this may help make code more concise
                raise ValueError(
                    "Should not pass `compression` with a `path`."
                )  # TODO: better message

        if array is not None:
            self.path = None
            self._array = array

            # TODO: validate compression?
            self._compression = compression

    @property
    def is_symbolic(self) -> bool:
        return self._array is None

    @property
    def array(self) -> np.ndarray:
        self.read()
        return self._array  # type: ignore

    @property
    def shape(self) -> Tuple[int, ...]:
        self.read()
        return self._array.shape  # type: ignore

    @property
    def dtype(self) -> str:
        self.read()
        return self._array.dtype.name  # type: ignore

    @property
    def compression(self) -> str:
        # TODO: raise exception if `read` wasn't called
        self.read()
        return self._compression.lower()

    def raw_bytes(self) -> bytes:
        if self.path is None:
            # TODO: get tobytes from meta
            return row_wise_to_bytes(self._array)

        else:
            with open(self.path, "rb") as image_file:
                return image_file.read()

    def read(self):
        if self._array is None:
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

                # TODO: validate compression?
                self._compression = img.format
                return self._array

            raise HubAutoUnsupportedFileExtensionError(self._suffix, SUPPORTED_SUFFIXES)

    def __str__(self):
        if self.is_symbolic:
            return f"Sample(is_symbolic=True, path={self.path})"

        return f"Sample(is_symbolic=False, shape={self.shape}, compression='{self.compression}', dtype='{self.dtype}' path={self.path})"

    def __repr__(self):
        return str(self)


def load(path: str) -> Sample:
    # TODO: mention that you can do `.numpy()` on this output to make it extremely easy to use
    return Sample(path)
