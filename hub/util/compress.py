from typing import Union
import numpy as np

from PIL import Image  # type: ignore
from io import BytesIO


SUPPORTED_COMPRESSIONS = ["png", "jpeg"]


def compress_array(array: np.ndarray, compression: str) -> bytes:
    if compression in SUPPORTED_COMPRESSIONS:
        img = Image.fromarray(array.astype("uint8"))
        out = BytesIO()
        img.save(out, compression)
        out.seek(0)
        return out.read()
    else:
        raise Exception()  # TODO


def decompress_array(buffer: Union[bytes, memoryview], compression: str) -> np.ndarray:
    if compression in SUPPORTED_COMPRESSIONS:
        # TODO: check if compression is actually the format (right now `compression` is useless)
        img = Image.open(BytesIO(buffer))
        return np.array(img)
    else:
        raise Exception()  # TODO
