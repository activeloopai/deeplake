from re import I
import numpy as np
from uuid import uuid4


CHUNK_ID_BITS = 64
CHUNK_NAME_ENCODING_DTYPE = np.uint64


# entry structure:
# [chunk_id, num_chunks, num_samples_per_chunk, last_index]

# index definitions:
CHUNK_ID_INDEX = 0
NUM_SAMPLES_INDEX = 1
LAST_INDEX_INDEX = 2


def _generate_chunk_id() -> CHUNK_NAME_ENCODING_DTYPE:
    return CHUNK_NAME_ENCODING_DTYPE(uuid4().int >> CHUNK_ID_BITS)


def _chunk_name_from_id(id: CHUNK_NAME_ENCODING_DTYPE) -> str:
    return hex(id)


class ChunkNameEncoder:
    def __init__(self):
        self._encoded = None
        self._connectivity = None

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

    def get_chunk_names(self, sample_index: int) -> CHUNK_NAME_ENCODING_DTYPE:
        """Returns the chunk names that correspond to `sample_index`."""

        if self.num_samples == 0:
            raise IndexError(
                f"Index {sample_index} is out of bounds for an empty chunk names encoding."
            )

        if sample_index < 0:
            sample_index = (self.num_samples) + sample_index

        idx = np.searchsorted(self._encoded[:, LAST_INDEX_INDEX], sample_index)
        id = self._encoded[idx, CHUNK_ID_INDEX]

        # TODO: return multiple ids
        return (_chunk_name_from_id(id),)

    def extend_chunk(self, num_samples: int, connected_to_next: bool = False):
        if num_samples <= 0:
            raise ValueError(
                f"When extending, `num_samples` should be > 0. Got {num_samples}."
            )

        if self.num_samples == 0:
            raise Exception(
                "Cannot extend the previous chunk because it doesn't exist."
            )

        if self._connectivity[-1] == 1:
            # TODO: custom exception
            raise Exception(
                "Cannot extend a chunk that is already marked as `connected_to_next`."
            )

        last_entry = self._encoded[-1]
        last_entry[LAST_INDEX_INDEX] += num_samples
        last_entry[NUM_SAMPLES_INDEX] += num_samples
        self._connectivity[-1] = connected_to_next

    def append_chunk(self, num_samples: int, connected_to_next: bool = False):
        if num_samples < 0:
            raise ValueError(
                f"When appending, `num_samples` should be >= 0. Got {num_samples}."
            )

        if self.num_samples == 0:
            if num_samples == 0:
                raise Exception("First num samples cannot be 0.")  # TODO: exceptions.py

            id = _generate_chunk_id()
            self._encoded = np.array(
                [[id, num_samples, num_samples - 1]], dtype=CHUNK_NAME_ENCODING_DTYPE
            )
            self._connectivity = np.array([connected_to_next], dtype=bool)
        else:
            if num_samples == 0 and self._connectivity[-1] == 0:
                raise Exception(
                    "num_samples cannot be 0 unless the previous chunk was connected to next."
                )  # TODO: exceptions.py

            id = _generate_chunk_id()

            # TODO: check if we can use the previous chunk name (and add the delimited range)
            last_index = self.num_samples - 1

            new_entry = np.array(
                [[id, num_samples, last_index + num_samples]],
                dtype=CHUNK_NAME_ENCODING_DTYPE,
            )
            self._encoded = np.concatenate([self._encoded, new_entry])
            self._connectivity = np.concatenate(
                [self._connectivity, [connected_to_next]]
            )


def _validate_num_samples(num_samples: int):
    if num_samples <= 0:
        raise ValueError(f"`num_count` should be > 0. Got {num_samples}.")
