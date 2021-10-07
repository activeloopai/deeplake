from typing import Tuple
from hub.core.compression import STRUCT_II, NATIVE_INT32, BaseCompressor
from PIL import Image  # type: ignore


class PNG(BaseCompressor):
    def verify(self):
        self.image.verify()
        return Image._conv_type_shape(self.image)

    def read_shape_and_dtype(self) -> Tuple[Tuple[int, ...], str]:
        """Reads shape and dtype of a png file from a file like object or file contents.
        If a file like object is provided, all of its contents are NOT loaded into memory."""

        self.bytesio_object.seek(16)  # type: ignore
        size = STRUCT_II.unpack(self.bytesio_object.read(8))[::-1]  # type: ignore
        bits, colors = self.bytesio_object.read(2)  # type: ignore

        # Get the number of channels and dtype based on bits and colors:
        if colors == 0:
            if bits == 1:
                typstr = "|b1"
            elif bits == 16:
                typstr = NATIVE_INT32
            else:
                typstr = "|u1"
            nlayers = None
        else:
            typstr = "|u1"
            if colors == 2:
                nlayers = 3
            elif colors == 3:
                nlayers = None
            elif colors == 4:
                if bits == 8:
                    nlayers = 2
                else:
                    nlayers = 4
            else:
                nlayers = 4
        shape = size if nlayers is None else size + (nlayers,)
        return shape, typstr  # type: ignore
