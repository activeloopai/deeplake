from typing import Any, Sequence
from hub.constants import ENCODING_DTYPE
import numpy as np

# TODO: refactor all encoders for common methods and uniform interface
class Encoder:
    last_index_index: int = -1

    def __init__(self, encoded=None):
        # TODO: docstring

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
        return int(self._encoded[-1, self.last_index_index] + 1)

    def num_samples_at(self, translated_index: int) -> int:
        # TODO: docstring

        lower_bound = 0
        if len(self._encoded) > 1 and translated_index > 0:
            lower_bound = self._encoded[translated_index - 1, self.last_index_index] + 1
        upper_bound = self._encoded[translated_index, self.last_index_index] + 1

        return int(upper_bound - lower_bound)

    def translate_index(self, local_sample_index: int) -> int:
        """Translates `local_sample_index` into which row it corresponds to inside the encoded state.

        Args:
            local_sample_index (int): Index representing a sample, binary searched over the "last index" column.

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
            self._encoded[:, self.last_index_index], local_sample_index
        )

    def register_samples(self, item: Any, num_samples: int):
        # TODO: docstring

        # TODO: optimize this

        self.validate_incoming_item(item, num_samples)

        if self.num_samples != 0:
            if self.combine_condition(item):
                last_index = self._encoded[-1, self.last_index_index]
                new_last_index = self.do_combine(last_index, num_samples)

                self._encoded[-1, self.last_index_index] = new_last_index

            else:
                decomposable = self.make_decomposable(item)

                last_index = self._encoded[-1, self.last_index_index]
                next_last_index = self.do_combine(last_index, num_samples)

                shape_entry = np.array(
                    [[*decomposable, next_last_index]], dtype=ENCODING_DTYPE
                )

                self._encoded = np.concatenate([self._encoded, shape_entry], axis=0)

        else:
            decomposable = self.make_decomposable(item)
            self._encoded = np.array(
                [[*decomposable, num_samples - 1]], dtype=ENCODING_DTYPE
            )

    def validate_incoming_item(self, _, num_samples: int):
        # TODO: docstring (also make these "_" functions)

        if num_samples <= 0:
            raise ValueError(f"`num_samples` should be > 0. Got: {num_samples}")

    def combine_condition(self, item: Any) -> bool:
        # TODO: docstring

        raise NotImplementedError

    def do_combine(self, last_index: ENCODING_DTYPE, num_samples: int):
        # TODO: docstring

        return last_index + num_samples

    def make_decomposable(self, item: Any) -> Sequence:
        # TODO: docstring

        return item

    def derive_value(self, row: np.ndarray) -> np.ndarray:
        # TODO: docstring

        raise NotImplementedError

    def __getitem__(
        self, local_sample_index: int, return_row_index: bool = False
    ) -> Any:
        # TODO: docstring

        row_index = self.translate_index(local_sample_index)
        value = self.derive_value(
            self._encoded[row_index], row_index, local_sample_index
        )

        if return_row_index:
            return value, row_index

        return value
