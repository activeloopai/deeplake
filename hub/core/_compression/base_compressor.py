from abc import ABC
from io import BytesIO
from typing import Optional, Tuple, Union
from PIL import Image  # type: ignore


class BaseCompressor(ABC):
    file = None
    _filename: Optional[str] = None
    _buffer: Optional[Union[bytes, memoryview]] = None
    _bytesio_object: Optional[BytesIO] = None
    _image: Image = None

    def __init__(self, item):
        # TODO: docstring + types, also common field variables

        if isinstance(item, BytesIO):
            self._bytesio_object = item
        elif isinstance(item, str):
            self._filename = item
        elif isinstance(item, (bytes, memoryview)):
            self._buffer = item
        elif hasattr(item, "read"):
            # TODO: maybe don't use hasattr?
            self.file = item
        else:
            raise TypeError(f"Don't support {type(item)} as input to compressors.")

    @property
    def bytesio_object(self) -> BytesIO:
        if self._bytesio_object is None:
            self._bytesio_object = BytesIO(self.buffer)

        return self._bytesio_object

    @property
    def buffer(self) -> Union[bytes, memoryview]:
        if self._buffer is None:
            # TODO: load buffer from file
            raise NotImplementedError

        return self._buffer

    @property
    def image(self) -> Image:
        if self._image is None:
            self._image = Image.open(self.buffer)
        return self._image

    def verify(self):
        # TODO: docstring + types
        pass

    def read_shape_and_dtype(self) -> Tuple[Tuple[int, ...], str]:
        # TODO: docstring + types
        pass
