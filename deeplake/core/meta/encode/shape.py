from hub.core.meta.encode.base_encoder import Encoder, LAST_SEEN_INDEX_COLUMN
from hub.constants import ENCODING_DTYPE
from typing import Tuple
from hub.core.storage.provider import StorageProvider
import numpy as np


class ShapeEncoder(Encoder):
    def _derive_value(self, row: np.ndarray, *_) -> Tuple:  # type: ignore
        return tuple(row[:LAST_SEEN_INDEX_COLUMN])

    @property
    def dimensionality(self) -> int:
        return len(self[0])

    def _validate_incoming_item(self, shape: Tuple[int], _):
        if self.num_samples > 0:
            last_shape = self[-1]  # TODO: optimize this

            if len(shape) != len(last_shape):
                raise ValueError(
                    f"All sample shapes in a tensor must have the same len(shape). Expected: {len(last_shape)} got: {len(shape)}."
                )

        super()._validate_incoming_item(shape, _)

    def _combine_condition(
        self, shape: Tuple[int], compare_row_index: int = -1
    ) -> bool:
        last_shape = self._derive_value(self._encoded[compare_row_index])
        return shape == last_shape
