import numpy as np

from PIL import Image  # type: ignore
from io import BytesIO


SUPPORTED_COMPRESSIONS = ["png", "jpeg"]


def compress_array(array: np.ndarray, compression: str) -> bytes:
    if compression in SUPPORTED_COMPRESSIONS:
        img = Image.fromarray(array.astype("uint8"), "RGB")
        out = BytesIO()
        img.save(out, compression)
        out.seek(0)
        return out.read()
    else:
        raise Exception()  # TODO


def decompress_array(buffer: memoryview, compression: str) -> np.ndarray:
    if compression in SUPPORTED_COMPRESSIONS:
        img = Image.open(BytesIO(buffer))
        return np.array(img)
    else:
        raise Exception()  # TODO
