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

    def _combine_condition(
        self, shape: Tuple[int], compare_row_index: int = -1
    ) -> bool:
        last_shape = self._derive_value(self._encoded[compare_row_index])
        return shape == last_shape

    def update_shape(self, local_sample_index: int, new_shape: Tuple[int]):
        # TODO: this function needs optimization

        encoded_index = self.translate_index(local_sample_index)

        if self[-1] == new_shape:
            return

        # TODO: if shapes don't match and there is only 1 sample for this row, all you need to do is update shape
        # TODO: if shapes don't match and there is > 1 sample for this row, you will need to create a new row

        raise NotImplementedError
