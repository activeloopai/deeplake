from hub.core.meta.encode.base_encoder import Encoder
from hub.constants import ENCODING_DTYPE
from typing import Tuple
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
            >>> enc.add_byte_position(4, 100)  # 100 int32 samples
            >>> enc._encoded_byte_position
            [[4, 0, 99]]
            >>> enc.add_byte_position(8, 100)
            >>> enc._encoded_byte_position
            [[4, 0, 99],
             [8, 400, 199]]
             >>> enc.add_byte_position(4, 200)
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
        """Calculates the amount of bytes total under a specific row. Useful for adding new rows to `_encoded`."""

        if len(self._encoded) == 0:
            return 0

        if row_index < 0:
            row_index = len(self._encoded) + row_index

        row = self._encoded[row_index]

        if row_index == 0:
            previous_last_index = -1
        else:
            previous_last_index = self._encoded[row_index - 1, self.last_index_index]

        num_samples = row[self.last_index_index] - previous_last_index
        num_bytes_for_entry = num_samples * row[NUM_BYTES_INDEX]
        return int(num_bytes_for_entry + row[START_BYTE_INDEX])

    @property
    def nbytes(self):
        if len(self._encoded) == 0:
            return 0
        return self._encoded.nbytes

    @property
    def array(self):
        return self._encoded

    def add_byte_position(self, num_bytes_per_sample: int, num_samples: int):
        """Either adds a new row to `_encoded`, or extends the last one. For more information on this algorithm, see `__init__`."""

        if num_samples <= 0:
            raise ValueError(f"`num_samples` should be > 0. Got {num_samples}.")

        if num_bytes_per_sample < 0:
            raise ValueError(f"`num_bytes` must be >= 0. Got {num_bytes_per_sample}.")

        if self.num_samples != 0:
            last_entry = self._encoded[-1]
            last_nb = last_entry[NUM_BYTES_INDEX]

            if last_nb == num_bytes_per_sample:
                self._encoded[-1, self.last_index_index] += num_samples

            else:
                last_index = last_entry[self.last_index_index]

                sb = self.num_bytes_encoded_under_row(-1)

                entry = np.array(
                    [[num_bytes_per_sample, sb, last_index + num_samples]],
                    dtype=ENCODING_DTYPE,
                )
                self._encoded = np.concatenate([self._encoded, entry], axis=0)

        else:
            self._encoded = np.array(
                [[num_bytes_per_sample, 0, num_samples - 1]],
                dtype=ENCODING_DTYPE,
            )

    def __getitem__(self, sample_index: int) -> Tuple[int, int]:
        """Get the (start_byte, end_byte) for `sample_index`. For details on the lookup algorithm, see `__init__`."""

        encoded_index = self.translate_index(sample_index)

        entry = self._encoded[encoded_index]

        index_bias = 0
        if encoded_index >= 1:
            index_bias = self._encoded[encoded_index - 1][self.last_index_index] + 1

        row_num_bytes = entry[NUM_BYTES_INDEX]
        row_start_byte = entry[START_BYTE_INDEX]

        start_byte = row_start_byte + (sample_index - index_bias) * row_num_bytes
        end_byte = start_byte + row_num_bytes
        return int(start_byte), int(end_byte)

    def update_num_bytes(self, local_sample_index: int, num_bytes: int):
        # TODO: docstring

        # TODO: this function needs optimization

        encoded_index = self.translate_index(local_sample_index)

        entry = self._encoded[encoded_index]

        if entry[NUM_BYTES_INDEX] == num_bytes:
            return

        # TODO: if num_bytes don't match and there is only 1 sample for this row, all you need to do is update num_bytes
        # TODO: if num_bytes don't match and there is > 1 sample for this row, you will need to create a new row

        raise NotImplementedError
