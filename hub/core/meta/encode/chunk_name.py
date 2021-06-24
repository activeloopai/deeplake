from re import I
import numpy as np
from uuid import uuid4


CHUNK_ID_BITS = 64
CHUNK_NAME_ENCODING_DTYPE = np.uint64


# entry structure:
# [chunk_id, num_chunks, num_samples_per_chunk, last_index]

# index definitions:
CHUNK_ID_INDEX = 0
NUM_CHUNKS_INDEX = 1
NUM_SAMPLES_INDEX = 2
LAST_INDEX_INDEX = 3


def _generate_chunk_id() -> CHUNK_NAME_ENCODING_DTYPE:
    return CHUNK_NAME_ENCODING_DTYPE(uuid4().int >> CHUNK_ID_BITS)


def _chunk_name_from_id(id: CHUNK_NAME_ENCODING_DTYPE) -> str:
    return hex(id)


class ChunkNameEncoder:
    def __init__(self):
        self._encoded = None

    @property
    def num_ids(self) -> int:
        if self._encoded is None:
            return 0
        return len(self._encoded)

    @property
    def num_samples(self) -> int:
        if self._encoded is None:
            return 0
        return int(self._encoded[-1, LAST_INDEX_INDEX] + 1)

    def get_chunk_id(self, sample_index: int) -> CHUNK_NAME_ENCODING_DTYPE:
        """Returns the chunk name corresponding to `sample_index`."""

        if self.num_samples == 0:
            raise IndexError(
                f"Index {sample_index} is out of bounds for an empty chunk names encoding."
            )

        if sample_index < 0:
            sample_index = (self.num_samples) + sample_index

        idx = np.searchsorted(self._encoded[:, LAST_INDEX_INDEX], sample_index)
        id = self._encoded[idx, CHUNK_ID_INDEX]
        # TODO: return actual chunk name (with delimited range)
        return _chunk_name_from_id(id)

    def extend_chunk(self, count: int):
        _validate_count(count)

        if self.num_samples == 0:
            raise Exception(
                "Cannot extend the previous chunk because it doesn't exist."
            )

        last_entry = self._encoded[-1]
        last_entry[LAST_INDEX_INDEX] += count
        last_entry[NUM_SAMPLES_INDEX] += count

        # TODO: check if previous chunk can be combined

    def append_chunk(self, count: int):
        _validate_count(count)

        if self.num_samples == 0:
            id = _generate_chunk_id()
            self._encoded = np.array(
                [[id, 1, count, count - 1]], dtype=CHUNK_NAME_ENCODING_DTYPE
            )
        else:
            self.try_combining_last_two_chunks()

            id = _generate_chunk_id()

            # TODO: check if we can use the previous chunk name (and add the delimited range)
            last_index = self.num_samples - 1

            new_entry = np.array(
                [[id, 1, count, last_index + count]], dtype=CHUNK_NAME_ENCODING_DTYPE
            )
            self._encoded = np.concatenate([self._encoded, new_entry])

        # TODO: check if previous chunk can be combined

    def try_combining_last_two_chunks(self) -> bool:
        # TODO: docstring that explains what this does

        if self.num_ids == 0:
            # TODO: exceptions.py
            raise Exception("Cannot finalize last chunk because it doesn't exist.")

        # can only combine if at least 2 unique chunk_ids exist
        if self.num_ids >= 2:
            last_entry = self._encoded[-2]
            current_entry = self._encoded[-1]

            can_combine = (
                current_entry[NUM_SAMPLES_INDEX] == last_entry[NUM_SAMPLES_INDEX]
            )

            if can_combine:
                last_entry[LAST_INDEX_INDEX] = current_entry[LAST_INDEX_INDEX]
                self._encoded = self._encoded[:-1]
                return True

        return False


def _validate_count(count: int):
    if count <= 0:
        raise ValueError(f"Sample `count` should be > 0. Got {count}.")
