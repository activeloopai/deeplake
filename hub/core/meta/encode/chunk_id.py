from hub.util.exceptions import ChunkIdEncoderError
import hub
from hub.core.storage.cachable import Cachable
from io import BytesIO
from typing import Optional, Tuple
import numpy as np
from uuid import uuid4


CHUNK_ID_INDEX = 0
LAST_INDEX_INDEX = 1


class ChunkIdEncoder(Cachable):
    def __init__(self):
        """Encodes chunk IDs such that they can be mapped to and from sample indices.

        Note:
            This map has a time complexity of `O(log(N))`, where `N` is the number of unique chunk IDs in this instance.
            If this encoder is sharded, the time complexity's `N` is the max number of unique chunk IDs allowed in a single shard
                at the worst case scenario.
        """

        self._encoded_ids = None
        self._encoded_connectivity = None

    def tobytes(self) -> memoryview:
        bio = BytesIO()
        np.savez(
            bio,
            version=hub.__encoded_version__,
            ids=self._encoded_ids,
            connectivity=self._encoded_connectivity,
        )
        return bio.getbuffer()

    @staticmethod
    def name_from_id(id: np.uint64) -> str:
        return hex(id)[2:]

    @staticmethod
    def id_from_name(name: str) -> np.uint64:
        return int("0x" + name, 16)

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

    def generate_chunk_id(self) -> np.uint64:
        """Generates a random 64bit chunk ID using uuid4. Also prepares this ID to have samples registered to it.
        This method should be called once per chunk created.

        Returns:
            np.uint64: The random chunk ID.
        """

        id = np.uint64(uuid4().int >> 64)  # `id` is 64 bits after right shift

        if self.num_samples == 0:
            self._encoded_ids = np.array([[id, -1]], dtype=np.uint64)
            self._encoded_connectivity = np.array([False], dtype=bool)

        else:
            last_index = self.num_samples - 1

            new_entry = np.array(
                [[id, last_index]],
                dtype=np.uint64,
            )
            self._encoded_ids = np.concatenate([self._encoded_ids, new_entry])
            self._encoded_connectivity = np.concatenate(
                [self._encoded_connectivity, [False]]
            )

        return id

    def register_samples_to_last_chunk_id(self, num_samples: int):
        """Registers samples to the chunk ID that was generated last with the `generate_chunk_id` method.
        This method should be called at least once per chunk created.

        Args:
            num_samples (int): The number of samples the last chunk ID should have added to it's registration.

        Raises:
            ValueError: `num_samples` should be non-negative.
            ChunkIdEncoderError: Must call `generate_chunk_id` before registering samples.
            ChunkIdEncoderError: `num_samples` can only be 0 if it is able to be a sample continuation accross chunks.
        """

        if num_samples < 0:
            raise ValueError(
                f"Cannot register negative num samples. Got: {num_samples}"
            )

        if self.num_samples == 0:
            raise ChunkIdEncoderError(
                "Cannot register samples because no chunk IDs exist."
            )

        if num_samples == 0 and self.num_chunks < 2:
            raise ChunkIdEncoderError(
                "Cannot register 0 num_samples (signifying a partial sample continuing the last chunk) when no last chunk exists."
            )

        current_entry = self._encoded_ids[-1]

        # this operation will trigger an overflow for the first addition, so supress the warning
        np.seterr(over="ignore")
        current_entry[LAST_INDEX_INDEX] += np.uint64(num_samples)
        np.seterr(over="warn")

    def register_connection_to_last_chunk_id(self):
        """The last generated chunk ID can be connected to the chunk ID that preceeds it. A connection means that they share a common sample.

        Raises:
            ChunkIdEncoderError: Connections require at least 2 chunk IDs to exist.
        """

        if self.num_chunks < 2:
            raise ChunkIdEncoderError(
                "Cannot register connection because at least 2 chunk ids need to exist. See: `generate_chunk_id`"
            )

        current_entry = self._encoded_ids[-2]
        self._encoded_connectivity[-2] = True

    def get_name_for_chunk(self, idx) -> str:
        return ChunkIdEncoder.name_from_id(self._encoded_ids[:, CHUNK_ID_INDEX][idx])

    def get_local_sample_index(self, global_sample_index: int) -> int:
        """Converts a global sample index into an index local to the chunk that it first appears in.

        Examples:
            Given: 5 samples. First 2 fit in chunk 0, 3rd sample is in chunk 0 and chunk 1, the rest is in chunk 1.

            >>> enc.num_chunks
            2
            >>> enc.num_samples
            5

            >>> enc.get_local_sample_index(0)
            0
            >>> enc.get_local_sample_index(1)
            1
            >>> enc.get_local_sample_index(2)
            0
            >>> enc.get_local_sample_index(3)
            1
            >>> enc.get_local_sample_index(4)
            2


        Args:
            global_sample_index (int): Integer index relative to the tensor.

        Raises:
            NotImplementedError: Doesn't support negative indexing.

        Returns:
            int: Integer index relative to the chunk that `global_sample_index` first appears in.
        """

        _, chunk_indices = self.__getitem__(global_sample_index, return_indices=True)
        chunk_index = chunk_indices[0]

        if global_sample_index < 0:
            raise NotImplementedError

        if chunk_index == 0:
            return global_sample_index

        current_entry = self._encoded_ids[chunk_index - 1]
        last_num_samples = current_entry[LAST_INDEX_INDEX] + 1

        return int(global_sample_index - last_num_samples)

    def __getitem__(
        self, sample_index: int, return_indices: bool = False
    ) -> Tuple[Tuple[np.uint64], Optional[Tuple[int]]]:
        """Get all chunks IDs where `sample_index` is contained.

        Args:
            sample_index (int): Sample index. May or may not span accross multiple chunks.
            return_indices (bool): If True, 2 tuples are returned. One with IDs and the other with the indices of those chunk IDs. Defaults to False.

        Raises:
            IndexError: Sample index should be accounted for with `register_samples_to_last_chunk_id`.

        Returns:
            Tuple[np.uint64], Optional[Tuple[int]]: Chunk IDs. If `return_indices` is True, the indices are also returned for those chunk IDs.
        """

        if self.num_samples == 0:
            raise IndexError(
                f"Index {sample_index} is out of bounds for an empty chunk names encoding."
            )

        if sample_index < 0:
            sample_index = (self.num_samples) + sample_index

        idx = np.searchsorted(self._encoded_ids[:, LAST_INDEX_INDEX], sample_index)
        ids = [self._encoded_ids[idx, CHUNK_ID_INDEX]]
        indices = [idx]

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
