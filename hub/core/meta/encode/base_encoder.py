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
    def nbytes(self):
        return self._encoded.nbytes

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

        # TODO: generalize all methods to this

        # TODO: optimize this

        if num_samples <= 0:
            raise ValueError(f"`num_samples` should be > 0. Got: {num_samples}")

        self.validate_incoming_item(item)

        if self.num_samples != 0:
            if self.combine_condition(item):
                self._encoded[-1, self.last_index_index] += num_samples

            else:
                decomposable = self.make_decomposable(item)

                last_index = self._encoded[-1, self.last_index_index]
                shape_entry = np.array(
                    [[*decomposable, last_index + num_samples]], dtype=ENCODING_DTYPE
                )

                self._encoded = np.concatenate([self._encoded, shape_entry], axis=0)

        else:
            decomposable = self.make_decomposable(item)
            self._encoded = np.array(
                [[*decomposable, num_samples - 1]], dtype=ENCODING_DTYPE
            )

    def validate_incoming_item(self, item: Any):
        # TODO: docstring

        raise NotImplementedError

    def combine_condition(self, item: Any) -> bool:
        # TODO: docstring

        raise NotImplementedError

    def make_decomposable(self, item: Any) -> Sequence:
        # TODO: docstring

        return item
