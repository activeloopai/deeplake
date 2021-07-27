from hub.core.meta.encode.base_encoder import Encoder, LAST_SEEN_INDEX_COLUMN
from hub.constants import ENCODING_DTYPE
from typing import Tuple
from hub.core.storage.provider import StorageProvider
import numpy as np


class ShapeEncoder(Encoder):
    def _derive_value(self, row: np.ndarray, *_) -> np.ndarray:
        return tuple(row[:LAST_SEEN_INDEX_COLUMN])

    def _validate_incoming_item(self, shape: Tuple[int], _):
        if len(self._encoded) > 0:
            last_shape = self[-1]  # TODO: optimize this

            if len(shape) != len(last_shape):
                raise ValueError(
                    f"All sample shapes in a tensor must have the same len(shape). Expected: {len(last_shape)} got: {len(shape)}."
                )

        super()._validate_incoming_item(shape, _)

    def _combine_condition(self, shape: Tuple[int]) -> bool:
        last_shape = self[-1]  # TODO: optimize this

        return shape == last_shape
