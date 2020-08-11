import numpy as np

from .base import Base


class Default(Base):
    def __init__(self):
        super().__init__()

    def encode(self, array: np.ndarray) -> bytes:
        with io.BytesIO() as f:
            array.save(f, allow_pickle=True)
            return f.getvalue()

    def decode(self, bytes_: bytes) -> np.ndarray:
        with io.BytesIO(bytes_) as f:
            return np.load(f, allow_pickle=True)
