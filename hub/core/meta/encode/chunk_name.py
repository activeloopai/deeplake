import numpy as np
from uuid import uuid4


CHUNK_ID_BITS = 64
CHUNK_NAME_ENCODING_DTYPE = np.uint64


def _generate_chunk_id() -> CHUNK_NAME_ENCODING_DTYPE:
    return CHUNK_NAME_ENCODING_DTYPE(uuid4().int >> CHUNK_ID_BITS)


def _chunk_name_from_id(id: CHUNK_NAME_ENCODING_DTYPE) -> str:
    return hex(id)


class ChunkNameEncoder:
    def __init__(self):
        self._encoded = None

    @property
    def num_samples(self) -> int:
        if self._encoded is None:
            return 0
        return int(self._encoded[-1, -1] + 1)

    def get_chunk_id(self, sample_index: int) -> CHUNK_NAME_ENCODING_DTYPE:
        """Returns the chunk name corresponding to `sample_index`."""

        if self.num_samples == 0:
            raise IndexError(
                f"Index {sample_index} is out of bounds for an empty chunk names encoding."
            )

        if sample_index < 0:
            sample_index = (self.num_samples) + sample_index

        idx = np.searchsorted(self._encoded[:, -1], sample_index)
        id = self._encoded[idx, 0]
        # TODO: return actual chunk name (with delimited range)
        return _chunk_name_from_id(id)

    def extend_chunk(self, count: int):
        _validate_count(count)

        if self.num_samples == 0:
            raise Exception(
                "Cannot extend the previous chunk because it doesn't exist."
            )

        self._encoded[-1, -1] += count

        # TODO: check if previous chunk can be combined

    def append_chunk(self, count: int):
        _validate_count(count)

        if self.num_samples == 0:
            id = _generate_chunk_id()
            self._encoded = np.array(
                [[id, 1, count - 1]], dtype=CHUNK_NAME_ENCODING_DTYPE
            )
        else:
            id = _generate_chunk_id()

            # TODO: check if we can use the previous chunk name (and add the delimited range)
            last_index = self.num_samples - 1

            new_entry = np.array(
                [[id, 1, last_index + count]], dtype=CHUNK_NAME_ENCODING_DTYPE
            )
            self._encoded = np.concatenate([self._encoded, new_entry])

        # TODO: check if previous chunk can be combined


def _validate_count(count: int):
    if count <= 0:
        raise ValueError(f"Sample `count` should be > 0. Got {count}.")
