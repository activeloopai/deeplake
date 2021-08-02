from abc import ABC
from typing import Any, Sequence
from hub.constants import ENCODING_DTYPE
import numpy as np


# the "last seen index" denotes an encoder row's last seen index
# this is the column that is binary searched over
LAST_SEEN_INDEX_COLUMN = -1


class Encoder(ABC):
    def __init__(self, encoded=None):
        """Base class for custom encoders that allow reading meta information from sample indices without decoding the entire encoded state.

        Handles heavy lifting logic for:
            - Chunk ID encoder
            - Shape encoder
            - Byte positions encoder

        Lookup algorithm is essentially the same for all encoders, however the details are different.
        You can find all of this information in their respective classes.

        Layout:
            `_encoded` is a 2D array.

            Best case scenario:
                The best case scenario is when all samples have the same meta and can thus be stored in a single row,
                providing a O(1) lookup.

            Worst case scenario:
                The worst case scenario is when every sample has different meta values. This means the number of rows is equal to the number
                of samples, providing a O(log(N)) lookup.

            Lookup algorithm:
                To get the decoded meta for some sample index, you do a binary search over the column `LAST_SEEN_INDEX_COLUMN`. This will give you
                the row that corresponds to that sample index (since the right-most column is our "last index" for that meta information).
                Then, you decode the row and that is your meta!

        Args:
            encoded (np.ndarray): Encoded state, if None state is empty. Helpful for deserialization. Defaults to None.
        """

        self._encoded = encoded
        if self._encoded is None:
            self._encoded = np.array([], dtype=ENCODING_DTYPE)

    @property
    def array(self):
        return self._encoded

    @property
    def nbytes(self):
        return self.array.nbytes

    @property
    def num_samples(self) -> int:
        if len(self._encoded) == 0:
            return 0
        return int(self._encoded[-1, LAST_SEEN_INDEX_COLUMN] + 1)

    def num_samples_at(self, translated_index: int) -> int:
        """Calculates the number of samples a row in the encoding corresponds to.

        Args:
            translated_index (int): This index will be used when indexing `self._encoded`.

        Returns:
            int: Representing the number of samples that a row's derivable value represents.
        """

        lower_bound = 0
        if len(self._encoded) > 1 and translated_index > 0:
            lower_bound = (
                self._encoded[translated_index - 1, LAST_SEEN_INDEX_COLUMN] + 1
            )
        upper_bound = self._encoded[translated_index, LAST_SEEN_INDEX_COLUMN] + 1

        return int(upper_bound - lower_bound)

    def translate_index(self, local_sample_index: int) -> int:
        """Searches for the row index for where `local_sample_index` exists within `self._encoded`.
        This method is worst case log(N) due to the binary search.

        Args:
            local_sample_index (int): Index representing a sample. Localized to `self._encoded`.

        Raises:
            IndexError: Cannot index when there are no samples to index into.

        Returns:
            int: The index of the corresponding row inside the encoded state.
        """

        # TODO: optimize this (should accept an optional argument for starting point, instead of random binsearch)

        if self.num_samples == 0:
            raise IndexError(
                f"Index {local_sample_index} is out of bounds for an empty byte position encoding."
            )

        if local_sample_index < 0:
            local_sample_index += self.num_samples

        return np.searchsorted(
            self._encoded[:, LAST_SEEN_INDEX_COLUMN], local_sample_index
        )

    def register_samples(self, item: Any, num_samples: int):
        """Register `num_samples` as `item`. Combines when the `self._combine_condition` returns True.
        This method adds data to `self._encoded` without decoding.

        Args:
            item (Any): General input, will be passed along to subclass methods.
            num_samples (int): Number of samples that have `item`'s value. Will be passed along to subclass methods.
        """

        # TODO: optimize this

        self._validate_incoming_item(item, num_samples)

        if self.num_samples != 0:
            if self._combine_condition(item):
                last_index = self._encoded[-1, LAST_SEEN_INDEX_COLUMN]
                new_last_index = self._derive_next_last_index(last_index, num_samples)

                self._encoded[-1, LAST_SEEN_INDEX_COLUMN] = new_last_index

            else:
                decomposable = self._make_decomposable(item)

                last_index = self._encoded[-1, LAST_SEEN_INDEX_COLUMN]
                next_last_index = self._derive_next_last_index(last_index, num_samples)

                shape_entry = np.array(
                    [[*decomposable, next_last_index]], dtype=ENCODING_DTYPE
                )

                self._encoded = np.concatenate([self._encoded, shape_entry], axis=0)

        else:
            decomposable = self._make_decomposable(item)
            self._encoded = np.array(
                [[*decomposable, num_samples - 1]], dtype=ENCODING_DTYPE
            )

    def _validate_incoming_item(self, item: Any, num_samples: int):
        """Raises appropriate exceptions for when `item` or `num_samples` are invalid.
        Subclasses should override this method when applicable.

        Args:
            item (Any): General input, will be passed along to subclass methods.
            num_samples (int): Number of samples that have `item`'s value. Will be passed along to subclass methods.

        Raises:
            ValueError: For the general case, `num_samples` should be > 0.
        """

        if num_samples <= 0:
            raise ValueError(f"`num_samples` should be > 0. Got: {num_samples}")

    def _combine_condition(self, item: Any) -> bool:
        """Should determine if `item` can be combined with the last row in `self._encoded`."""

    def _derive_next_last_index(self, last_index: ENCODING_DTYPE, num_samples: int):
        """Calculates what the next last index should be."""
        return last_index + num_samples

    def _make_decomposable(self, item: Any) -> Sequence:
        """Should return a value that can be decompsed with the `*` operator. Example: `*(1, 2)`"""

        return item

    def _derive_value(
        self, row: np.ndarray, row_index: int, local_sample_index: int
    ) -> np.ndarray:
        """Given a row of `self._encoded`, this method should implement how `__getitem__` hands a value to the caller."""

    def __getitem__(
        self, local_sample_index: int, return_row_index: bool = False
    ) -> Any:
        """Derives the value at `local_sample_index`.

        Args:
            local_sample_index (int): Index of the sample for the desired value.
            return_row_index (bool): If True, the index of the row that the value was derived from is returned as well.
                Defaults to False.

        Returns:
            Any: Either just a singular derived value, or a tuple with the derived value and the row index respectively.
        """

        row_index = self.translate_index(local_sample_index)
        value = self._derive_value(
            self._encoded[row_index], row_index, local_sample_index
        )

        if return_row_index:
            return value, row_index

        return value
