from hub.core.meta.encode.base_encoder import Encoder, LAST_SEEN_INDEX_COLUMN
from typing import Sequence
import numpy as np


NUM_BYTES_COLUMN = 0
START_BYTE_COLUMN = 1


class BytePositionsEncoder(Encoder):
    def num_bytes_encoded_under_row(self, row_index: int) -> int:
        """Calculates the amount of bytes total under a specific row. Useful for adding new rows to `_encoded`."""

        if len(self._encoded) == 0:
            return 0

        if row_index < 0:
            row_index = len(self._encoded) + row_index

        row = self._encoded[row_index]

        if row_index == 0:
            previous_last_index = -1
        else:
            previous_last_index = self._encoded[row_index - 1, LAST_SEEN_INDEX_COLUMN]

        num_samples = row[LAST_SEEN_INDEX_COLUMN] - previous_last_index
        num_bytes_for_entry = num_samples * row[NUM_BYTES_COLUMN]
        return int(num_bytes_for_entry + row[START_BYTE_COLUMN])

    def _validate_incoming_item(self, num_bytes: int, _):
        if num_bytes < 0:
            raise ValueError(f"`num_bytes` must be >= 0. Got {num_bytes}.")

        super()._validate_incoming_item(num_bytes, _)

    def _combine_condition(self, num_bytes: int) -> bool:
        last_num_bytes = self._encoded[-1, NUM_BYTES_COLUMN]
        return num_bytes == last_num_bytes

    def _make_decomposable(self, num_bytes: int) -> Sequence:
        sb = self.num_bytes_encoded_under_row(-1)
        return [num_bytes, sb]

    def _derive_value(
        self, row: np.ndarray, row_index: int, local_sample_index: int
    ) -> np.ndarray:
        index_bias = 0
        if row_index >= 1:
            index_bias = self._encoded[row_index - 1][LAST_SEEN_INDEX_COLUMN] + 1

        row_num_bytes = row[NUM_BYTES_COLUMN]
        row_start_byte = row[START_BYTE_COLUMN]

        start_byte = row_start_byte + (local_sample_index - index_bias) * row_num_bytes
        end_byte = start_byte + row_num_bytes
        return int(start_byte), int(end_byte)
