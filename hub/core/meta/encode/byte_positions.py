from hub.core.meta.encode.base_encoder import Encoder, LAST_SEEN_INDEX_INDEX
from hub.constants import ENCODING_DTYPE
from typing import Sequence, Tuple
import numpy as np


# these constants are for accessing the data layout. see the `BytePositionsEncoder` docstring.
NUM_BYTES_INDEX = 0
START_BYTE_INDEX = 1


class BytePositionsEncoder(Encoder):
    """Custom compressor that allows reading of byte positions from a sample index without decompressing.

    Byte Positions:
        Byte positions are 2 numbers: `start_byte` and `end_byte`. This represents where in the chunk's `data` buffer
        a sample lives.
        Byte positions are encoded as `num_bytes` and `start_byte`. That way we can group together samples that have the same
        `num_bytes`.

    Layout:
        `_encoded` is a 2D array.

        Rows:
            The number of rows is equal to the number of unique runs of `num_bytes` that exist upon ingestion. See examples below.

        Columns:
            The number of columns is 3.
            Each row looks like this: [num_bytes, start_byte, last_index], where `last_index` is the last index that a sample has the same
            `num_bytes` in a run.

        Example:
            >>> enc = BytePositionsEncoder()
            >>> enc.register_samples(4, 100)  # 100 int32 samples
            >>> enc._encoded_byte_position
            [[4, 0, 99]]
            >>> enc.register_samples(8, 100)
            >>> enc._encoded_byte_position
            [[4, 0, 99],
             [8, 400, 199]]
             >>> enc.register_samples(4, 200)
            >>> enc._encoded_byte_position
            [[4, 0, 99],
             [8, 400, 199],
             [4, 1200, 399]]
            >>> enc.num_samples
            400

        Best case scenario:
            The best case scenario is when all samples are the same number of bytes long. This means the number of rows is 1,
            providing a O(1) lookup.

        Worst case scenario:
            The worst case scenario is when all samples are a differet number of bytes long. This means the number of rows is equal to the number
            of samples, providing a O(log(N)) lookup.

        Lookup algorithm:
            To get the byte position for some sample index, you do a binary search over the right-most column. This will give you
            the row that corresponds to that sample index (since the right-most column is our "last index" for that byte position).
            Then, you get the `num_bytes` and `start_byte` from that row. You can now derive, using `last_index` what the `start_byte`
            and `end_byte` are exactly.

            derive byte position for `index` after finding the row:
                row_start_byte = this row's stored `start_byte`.
                index_bias = previous row's stored `start_byte` if it exists. If it doesn't, use 0.
                start_byte = row_start_byte + (index - index_bias) * row_num_bytes
                end_byte = start_byte + row_num_bytes
                byte_position = (start_byte, end_byte)
    """

    def num_bytes_encoded_under_row(self, row_index: int) -> int:
        """Calculates the amount of bytes total "under" a specific row. "Under" meaning all rows that preceed it. Useful for adding new rows to `_encoded`."""

        if len(self._encoded) == 0 or row_index == 0:
            return 0

        if row_index < 0:
            row_index = len(self._encoded) + row_index

        row = self._encoded[row_index]

        if row_index == 0:
            previous_last_index = -1
        else:
            previous_last_index = self._encoded[row_index - 1, LAST_SEEN_INDEX_INDEX]

        num_samples = row[LAST_SEEN_INDEX_INDEX] - previous_last_index
        num_bytes_for_entry = num_samples * row[NUM_BYTES_INDEX]
        return int(num_bytes_for_entry + row[START_BYTE_INDEX])

    def _validate_incoming_item(self, num_bytes: int, _):
        if num_bytes < 0:
            raise ValueError(f"`num_bytes` must be >= 0. Got {num_bytes}.")

        super()._validate_incoming_item(num_bytes, _)

    def _combine_condition(self, num_bytes: int, compare_row_index: int = -1) -> bool:
        """Checks if `num_bytes` matches the `num_bytes` represented at row with index `compare_row_index`."""

        last_num_bytes = self._encoded[compare_row_index, NUM_BYTES_INDEX]
        return num_bytes == last_num_bytes

    def _make_decomposable(
        self, num_bytes: int, compare_row_index: int = -1
    ) -> Sequence:
        sb = self.num_bytes_encoded_under_row(compare_row_index)
        return [num_bytes, sb]

    def _derive_value(
        self, row: np.ndarray, row_index: int, local_sample_index: int
    ) -> np.ndarray:
        index_bias = 0
        if row_index >= 1:
            index_bias = self._encoded[row_index - 1][LAST_SEEN_INDEX_INDEX] + 1

        row_num_bytes = row[NUM_BYTES_INDEX]
        row_start_byte = row[START_BYTE_INDEX]

        start_byte = row_start_byte + (local_sample_index - index_bias) * row_num_bytes
        end_byte = start_byte + row_num_bytes
        return int(start_byte), int(end_byte)

    # def update_num_bytes(self, local_sample_index: int, num_bytes: int):
    #     # TODO: docstring

    #     # TODO: GENERALIZE INTO BASE CLASS!!!

    #     # TODO: this function needs optimization

    #     row_index = self.translate_index(local_sample_index)

    #     if self._combine_condition(num_bytes, row_index):
    #         # num_bytes matches the current row's num_bytes (no need to make any changes)

    #         # TODO: TEST THIS!
    #         return

    #     if self._combine_condition(num_bytes, row_index + 1):
    #         # num_bytes matches the next row's num_bytes (no need to create a new row)

    #         # TODO: TEST THIS!

    #         # TODO: decrement num_samples by 1 at `row_index`
    #         # TODO: then increment num_samples by 1 at `row_index + 1`

    #         raise NotImplementedError

    #     if self.num_samples_at(row_index) == 1:
    #         # TODO: TEST THIS!

    #         # TODO: if num_bytes don't match and there is only 1 sample for this row, all you need to do is update num_bytes
    #         raise NotImplementedError

    #     # at this point, `num_bytes` doesn't match and there are > 1 samples at `row_index`.

    #     # TODO: create a new row that represents a single sample with `num_bytes` as it's derived value
    #     # TODO: decrement current row's last_index_index

    #     # TODO: do something with `upper_encoded`
    #     lower_rows = self._encoded[:row_index]

    #     last_index = (
    #         0 if len(lower_rows) == 0 else lower_rows[-1, self.last_index_index]
    #     )

    #     new_item = self._make_decomposable(num_bytes, compare_row_index=row_index)
    #     new_row = np.array([*new_item, last_index], dtype=ENCODING_DTYPE)

    #     upper_rows = self._encoded[row_index:]

    #     # TODO: if num_bytes don't match and there is > 1 sample for this row, you will need to create a new row

    #     # TODO: memcp (remove for optimization)
    #     self._encoded = np.concatenate([lower_rows, [new_row], upper_rows])

    #     # TODO: update following row's start byte
