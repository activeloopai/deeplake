from typing import Tuple

import hub.features.tensor as tensor


class Image(tensor.Tensor):
    def __init__(
        shape: Tuple[int, ...] = None,
        dtype=None,
        encoding_format: str = None,
        channels=None,
    ):
        super().__init__(shape, dtype)

    @property
    def encoding_format(self):
        raise NotImplementedError()
