from abc import ABC
from io import BufferedReader, BytesIO
from typing import Optional, Tuple, Union
from PIL import Image  # type: ignore


class BaseCompressor(ABC):
    file = None
    _filename: Optional[str] = None
    _buffer: Optional[Union[bytes, memoryview]] = None
    _bytesio_object: Optional[BytesIO] = None
    _buffered_reader_object: Optional[BufferedReader] = None
    _image: Image = None

    def __init__(self, item):
        if isinstance(item, BytesIO):
            self._bytesio_object = item
        elif isinstance(item, str):
            self._filename = item
        elif isinstance(item, (bytes, memoryview)):
            self._buffer = item
        elif isinstance(item, BufferedReader):
            self._buffered_reader_object = item
        elif hasattr(item, "read"):
            # TODO: maybe don't use hasattr(item, "read")?
            self.file = item
        else:
            raise TypeError(f"Don't support {type(item)} as input to compressors.")

    @property
    def buffer(self) -> Union[bytes, memoryview]:
        if self._buffer is None:
            if self._buffered_reader_object is not None:
                self._buffer = self._buffered_reader_object.read()

            elif self._filename is not None:
                self._buffer = open(self._filename, "rb").read()

            else:
                raise NotImplementedError

        # from mp3 edge case
        if isinstance(self._buffer, memoryview):
            if (
                isinstance(self._buffer.obj, bytes)
                and self._buffer.strides == (1,)
                and self._buffer.shape == (len(self._buffer.obj),)
            ):
                self._buffer = self._buffer.obj
            else:
                self._buffer = bytes(self._buffer)

        return self._buffer

    @property
    def bytesio_object(self) -> BytesIO:
        if self._bytesio_object is None:
            self._bytesio_object = BytesIO(self.buffer)

        self._bytesio_object.seek(0)
        return self._bytesio_object

    @property
    def image(self) -> Image:
        if self._image is None:
            if self.file is not None:
                self._image = Image.open(self.file)
            else:
                self._image = Image.open(self.bytesio_object)
        return self._image

    def verify(self):
        pass

    def read_shape_and_dtype(self) -> Tuple[Tuple[int, ...], str]:
        pass

    def decompress(self):
        pass
