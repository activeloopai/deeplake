from typing import Tuple

import hub.features.tensor as tensor


class Segmentation(tensor.Tensor):
    def __init__(
        self,
        shape: Tuple[int, ...] = None,
        dtype: str = None,
        num_classes: int = None,
        names: Tuple[str] = None,
        names_file: str = None,
    ):
        pass
