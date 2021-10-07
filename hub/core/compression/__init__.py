import sys
import struct
import re


if sys.byteorder == "little":
    NATIVE_INT32 = "<i4"
    NATIVE_FLOAT32 = "<f4"
else:
    NATIVE_INT32 = ">i4"
    NATIVE_FLOAT32 = ">f4"

STRUCT_HHB = struct.Struct(">HHB")
STRUCT_II = struct.Struct(">ii")


def re_find_first(pattern, string):
    for match in re.finditer(pattern, string):
        return match


from .base_compressor import BaseCompressor  # type: ignore
from .image.jpeg import JPEG  # type: ignore
from .image.png import PNG  # type: ignore
from .audio.audio import Audio  # type: ignore
