from hub.core.storage.cachable import Cachable
from io import BytesIO
from typing import Tuple
import numpy as np
from uuid import uuid4


CHUNK_ID_BITS = 64
CHUNK_NAME_ENCODING_DTYPE = np.uint64


# entry structure:
# [chunk_id, num_chunks, num_samples_per_chunk, last_index]

# index definitions:
CHUNK_ID_INDEX = 0
LAST_INDEX_INDEX = 1


def chunk_name_from_id(id: CHUNK_NAME_ENCODING_DTYPE) -> str:
    return hex(id)[2:]


def chunk_id_from_name(name: str) -> CHUNK_NAME_ENCODING_DTYPE:
    return int("0x" + name, 16)


class ChunkIdEncoder(Cachable):
    def __init__(self):
        self._encoded_ids = None
        self._encoded_connectivity = None

    def tobytes(self) -> memoryview:
        bio = BytesIO()
        np.savez(bio, ids=self._encoded_ids, connectivity=self._encoded_connectivity)
        return bio.getbuffer()

    @classmethod
    def frombuffer(cls, buffer: bytes):
        instance = cls()
        bio = BytesIO(buffer)
        npz = np.load(bio)
        instance._encoded_ids = npz["ids"]
        instance._encoded_connectivity = npz["connectivity"]
        return instance

    @property
    def num_chunks(self) -> int:
        if self._encoded_ids is None:
            return 0
        return len(self._encoded_ids)

    @property
    def num_samples(self) -> int:
        if self._encoded_ids is None:
            return 0
        return int(self._encoded_ids[-1, LAST_INDEX_INDEX] + 1)

    @property
    def last_chunk_name(self) -> str:
        return self.get_name_for_chunk(-1)

    def generate_chunk_id(self) -> CHUNK_NAME_ENCODING_DTYPE:
        id = CHUNK_NAME_ENCODING_DTYPE(uuid4().int >> CHUNK_ID_BITS)

        if self.num_samples == 0:
            self._encoded_ids = np.array([[id, 0]], dtype=CHUNK_NAME_ENCODING_DTYPE)
            self._encoded_connectivity = np.array([False], dtype=bool)
        else:
            last_index = self.num_samples - 1

            new_entry = np.array(
                [[id, last_index]],
                dtype=CHUNK_NAME_ENCODING_DTYPE,
            )
            self._encoded_ids = np.concatenate([self._encoded_ids, new_entry])
            self._encoded_connectivity = np.concatenate(
                [self._encoded_connectivity, [False]]
            )

        return id

    def register_samples_to_last_chunk_id(self, num_samples: int):
        if num_samples <= 0:
            raise ValueError(
                f"When extending, `num_samples` should be > 0. Got {num_samples}."
            )

        if self.num_samples == 0:
            raise Exception(
                "Cannot extend the previous chunk because it doesn't exist."
            )

        last_entry = self._encoded_ids[-1]
        last_entry[LAST_INDEX_INDEX] += CHUNK_NAME_ENCODING_DTYPE(num_samples)

    def get_name_for_chunk(self, idx) -> str:
        return chunk_name_from_id(self._encoded_ids[:, CHUNK_ID_INDEX][idx])

    def get_local_sample_index(self, global_sample_index: int):
        # TODO: explain what's going on here

        _, chunk_index = self.__getitem__(
            global_sample_index, return_indices=True, first_only=True
        )

        if global_sample_index < 0:
            raise Exception()  # TODO

        if chunk_index == 0:
            return global_sample_index

        last_entry = self._encoded_ids[chunk_index - 1]
        last_num_samples = last_entry[LAST_INDEX_INDEX] + 1

        return int(global_sample_index - last_num_samples)

    def __getitem__(
        self, sample_index: int, return_indices: bool = False, first_only: bool = False
    ):
        """Returns the chunk names that correspond to `sample_index`."""
        # TODO: docstring

        if self.num_samples == 0:
            raise IndexError(
                f"Index {sample_index} is out of bounds for an empty chunk names encoding."
            )

        if sample_index < 0:
            sample_index = (self.num_samples) + sample_index

        idx = np.searchsorted(self._encoded_ids[:, LAST_INDEX_INDEX], sample_index)
        ids = [self._encoded_ids[idx, CHUNK_ID_INDEX]]
        indices = [idx]

        if first_only:
            if return_indices:
                return ids[0], idx

            return ids[0]

        # if accessing last index, check connectivity!
        while (
            self._encoded_ids[idx, LAST_INDEX_INDEX] == sample_index
            and self._encoded_connectivity[idx]
        ):
            idx += 1
            name = self._encoded_ids[idx, CHUNK_ID_INDEX]
            ids.append(name)
            indices.append(idx)

        if return_indices:
            return tuple(ids), tuple(indices)

        return tuple(ids)

    def attach_samples_to_last_chunk(
        self, num_samples: int, connected_to_next: bool = False
    ) -> str:

        if self._encoded_connectivity[-1] == 1:
            # TODO: custom exception
            raise Exception(
                "Cannot extend a chunk that is already marked as `connected_to_next`."
            )

        self._encoded_connectivity[-1] = connected_to_next

        return chunk_name_from_id(last_entry[CHUNK_ID_INDEX])


def _validate_num_samples(num_samples: int):
    if num_samples <= 0:
        raise ValueError(f"`num_count` should be > 0. Got {num_samples}.")
