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

    def decode(self, content: bytes) -> np.ndarray:
        raise NotImplementedError("Not ready yet")
