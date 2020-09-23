from typing import Tuple


class FlatTensor:
    def __init__(
        self, path: str, shape: Tuple[int, ...], max_shape: Tuple[int, ...], dtype: str
    ):
        self.path = path
        self.shape = shape
        self.max_shape = max_shape
        self.dtype = dtype
