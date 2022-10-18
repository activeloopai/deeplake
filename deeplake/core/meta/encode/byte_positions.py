from deeplake.core.meta.encode.base_encoder import Encoder, LAST_SEEN_INDEX_COLUMN

from typing import Optional, Sequence
import numpy as np


NUM_BYTES_COLUMN = 0
START_BYTE_COLUMN = 1


class BytePositionsEncoder(Encoder):
    def get_sum_of_bytes(self, until_row_index: int = -1) -> int:
        """Get the total number of bytes that are accounted for.
        This operation is O(1).

        Args:
            until_row_index (int): Optionally provide a row index for which the sum will end.
                Gets all bytes accounted for up until `until_row_index`.

        Returns:
            int: Number of bytes encoded at and below `until_row_index`.
        """

        if len(self._encoded) == 0:
            return 0

        if until_row_index < 0:
            until_row_index = len(self._encoded) + until_row_index

        last_last_seen_index = 0
        if until_row_index > 0:
            last_last_seen_index = self._encoded[
                until_row_index - 1, LAST_SEEN_INDEX_COLUMN
            ]

        row = self._encoded[until_row_index]
        start_byte = row[START_BYTE_COLUMN]
        num_bytes = row[NUM_BYTES_COLUMN]
        delta = int(row[LAST_SEEN_INDEX_COLUMN]) - int(last_last_seen_index)

        if until_row_index == 0:
            delta += 1

        return start_byte + (num_bytes * delta)

    def _combine_condition(self, num_bytes: int, compare_row_index: int = -1) -> bool:
        """Checks if `num_bytes` matches the `num_bytes` represented at row with index `compare_row_index`."""

        last_num_bytes = self._encoded[compare_row_index, NUM_BYTES_COLUMN]
        return num_bytes == last_num_bytes

    def _make_decomposable(
        self, num_bytes: int, compare_row_index: int = -1
    ) -> Sequence:
        """Used for updating. Return value is a sequence representing the row that can be decomposed using the `*` operator."""

        start_byte = self.get_sum_of_bytes(compare_row_index)
        return [num_bytes, start_byte]

    def _post_process_state(self, start_row_index: int):
        """Starting at `start_row_index`, move downwards through `self._encoded` and update all start bytes
        for each row if applicable. Used for updating."""

        for row_index in range(start_row_index, len(self._encoded)):
            if row_index == 0:
                bytes_under_row = 0
            else:
                bytes_under_row = self.get_sum_of_bytes(row_index - 1)
            self._encoded[row_index, START_BYTE_COLUMN] = bytes_under_row

    def _derive_value(self, row: np.ndarray, row_index: int, local_sample_index: int):
        index_bias = 0
        if row_index >= 1:
            index_bias = self._encoded[row_index - 1][LAST_SEEN_INDEX_COLUMN] + 1

        row_num_bytes = row[NUM_BYTES_COLUMN]
        row_start_byte = row[START_BYTE_COLUMN]

        start_byte = row_start_byte + (local_sample_index - index_bias) * row_num_bytes
        end_byte = start_byte + row_num_bytes
        return int(start_byte), int(end_byte)

    def pop(self, index: Optional[int] = None):
        if index is None:
            index = self.get_last_index_for_pop()
        (sb, eb), row = self.__getitem__(index, return_row_index=True)  # type: ignore
        prev = -1 if row == 0 else self._encoded[row - 1, LAST_SEEN_INDEX_COLUMN]
        num_samples_in_row = self._encoded[row, LAST_SEEN_INDEX_COLUMN] - prev
        num_bytes = eb - sb
        if num_samples_in_row == 0:
            raise ValueError("No samples to pop")
        self._encoded[row:, LAST_SEEN_INDEX_COLUMN] -= 1
        self._encoded[row + 1 :, START_BYTE_COLUMN] -= num_bytes

        # after subtracting 1, the row is now empty
        if num_samples_in_row == 1:
            self._encoded = np.delete(self._encoded, row, axis=0)
