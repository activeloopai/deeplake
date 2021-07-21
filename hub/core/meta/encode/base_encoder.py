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
    def num_samples(self) -> int:
        if len(self._encoded) == 0:
            return 0
        return int(self._encoded[-1, self.last_index_index] + 1)

    def translate_index(self, local_sample_index: int) -> int:
        """Translates `local_sample_index` into which row it corresponds to inside the encoded state.

        Args:
            local_sample_index (int): Index representing a sample, binary searched over the "last index" column.

        Raises:
            IndexError: Cannot index when there are no samples to index into.

        Returns:
            int: The index of the corresponding row inside the encoded state.
        """

        if self.num_samples == 0:
            raise IndexError(
                f"Index {local_sample_index} is out of bounds for an empty byte position encoding."
            )

        if local_sample_index < 0:
            local_sample_index += self.num_samples

        return np.searchsorted(
            self._encoded[:, self.last_index_index], local_sample_index
        )
