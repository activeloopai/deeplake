import io
import time
import random
import traceback
import json

import numpy as np
from PIL import Image as PILImage

from .base import Base


class Image(Base):
    def __init__(self, format: str):
        self._format = format
        super().__init__()

    def encode(self, array: np.ndarray) -> bytes:
        raise NotImplementedError("Not ready yet")
        info = {"shape": array.shape, "dtype": str(array.dtype)}

        assert array.dtype == "uint8"
        assert len(array.shape) >= 3
        assert array.shape[-1] == 3

        if len(array.shape) == 3:
            with io.BytesIO() as bs:
                img = PILImage.fromarray(array)
                img.save(bs, format=self._format)
                return bs.getvalue()
        else:
            images = []
            for i, index in enumerate(np.ndindex(array.shape[:-3])):
                bs = io.BytesIO()
                img = PILImage.fromarray(array[index])
                img.save(bs, format=self._format)
                images.append(bs.getvalue())
        info["images"] = images
        return msgpack.dumps(info)

    def decode(self, content: bytes) -> np.ndarray:
        raise NotImplementedError("Not ready yet")
        info = msgpack.loads(content)
        if "image" in info:
            img = PILImage.open(io.BytesIO(bytearray(info["image"])))
            img.reshape(info["shape"])
            return img
        else:
            array = np.zeros(info["shape"], info["dtype"])
            images = info["images"]
            for i, index in enumerate(np.ndindex(info["shape"][:-3])):
                img = PILImage.open(io.BytesIO(bytearray(images[i])))
                arr = np.asarray(img)
                array[index] = arr
            return array

