from typing import Any, List, Tuple, Union

from hub.util.chunks import chunk_name_from_id, random_chunk_id
from hub.core.meta.encode.base_encoder import Encoder, LAST_SEEN_INDEX_COLUMN
from hub.constants import ENCODING_DTYPE
from hub.util.exceptions import ChunkIdEncoderError
from hub.core.storage.cachable import Cachable
import numpy as np
from hub.core.serialize import serialize_chunkids, deserialize_chunkids


CHUNK_ID_COLUMN = 0


class ChunkIdEncoder(Encoder, Cachable):
    def tobytes(self) -> memoryview:
        return serialize_chunkids(self.version, [self._encoded])

    def get_name_for_chunk(self, chunk_index: int) -> str:
        """Gets the name for the chunk at index `chunk_index`. If you need to get the name for a chunk from a sample index, instead
        use `__getitem__`, then `chunk_name_from_id`."""

        chunk_id = self._encoded[:, CHUNK_ID_COLUMN][chunk_index]
        return chunk_name_from_id(chunk_id)

    @classmethod
    def frombuffer(cls, buffer: bytes):
        instance = cls()
        if not buffer:
            return instance
        version, ids = deserialize_chunkids(buffer)
        if ids.nbytes:
            instance._encoded = ids
        instance.version = version
        return instance

    @property
    def num_chunks(self) -> int:
        if self.num_samples == 0:
            return 0
        return len(self._encoded)


    def generate_chunk_id(self, id: ENCODING_DTYPE=None) -> ENCODING_DTYPE:
        """Generates a random 64bit chunk ID using uuid4. Also prepares this ID to have samples registered to it.
        This method should be called once per chunk created.

        Args:
            id (ENCODING_DTYPE): If None, a random chunk ID will be generated.

        Returns:
            ENCODING_DTYPE: The random chunk ID.
        """

        if id is None:
            id = random_chunk_id()

        if self.num_samples == 0:
            self._encoded = np.array([[id, -1]], dtype=ENCODING_DTYPE)

        else:
            last_index = self.num_samples - 1

            new_entry = np.array(
                [[id, last_index]],
                dtype=ENCODING_DTYPE,
            )
            self._encoded = np.concatenate([self._encoded, new_entry])

        return id

    def register_samples(self, num_samples: int):  # type: ignore
        """Registers samples to the chunk ID that was generated last with the `generate_chunk_id` method.
        This method should be called at least once per chunk created.

        Args:
            num_samples (int): The number of samples the last chunk ID should have added to it's registration.

        Raises:
            ValueError: `num_samples` should be non-negative.
            IndexError: At least 1 chunk is required for `num_samples` to be 0.
        """

        if num_samples < 0:
            raise ValueError(
                f"Can only register a positive number of samples. Got {num_samples}"
            )

        if num_samples == 0:
            super().register_samples(None, 1)
            if self.num_chunks <= 0:
                raise IndexError(
                    "Cannot register a chunk ID with 0 samples because no root chunk exists."
                )
            self._encoded[-1, -1] -= 1
        else:
            super().register_samples(None, num_samples)

    def translate_index_relative_to_chunks(self, global_sample_index: int) -> int:
        """Converts `global_sample_index` into a new index that is relative to the chunk the sample belongs to.

        Example:
            Given: 2 sampes in chunk 0, 2 samples in chunk 1, and 3 samples in chunk 2.
            >>> self.num_samples
            7
            >>> self.num_chunks
            3
            >>> self.translate_index_relative_to_chunks(0)
            0
            >>> self.translate_index_relative_to_chunks(1)
            1
            >>> self.translate_index_relative_to_chunks(2)
            0
            >>> self.translate_index_relative_to_chunks(3)
            1
            >>> self.translate_index_relative_to_chunks(6)
            2

        Args:
            global_sample_index (int): Index of the sample relative to the containing tensor.

        Returns:
            int: local index value between 0 and the amount of samples the chunk contains - 1.
        """

        _, chunk_index = self.__getitem__(global_sample_index, return_row_index=True)  # type: ignore

        if chunk_index == 0:
            return global_sample_index

        current_entry = self._encoded[chunk_index - 1]  # type: ignore
        last_num_samples = current_entry[LAST_SEEN_INDEX_COLUMN] + 1

        return int(global_sample_index - last_num_samples)

    def _validate_incoming_item(self, _, num_samples: int):
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

        # note: do not call super() method (num_samples can be 0)

    def _derive_next_last_index(self, last_index: ENCODING_DTYPE, num_samples: int):
        # this operation will trigger an overflow for the first addition, so supress the warning
        np.seterr(over="ignore")
        new_last_index = last_index + ENCODING_DTYPE(num_samples)
        np.seterr(over="warn")

        return new_last_index

    def _combine_condition(self, *args) -> bool:
        """Always returns True because sample registration can always be done. Used in base encoder `register_samples`."""

        return True

    def _derive_value(self, row: np.ndarray, *_) -> np.ndarray:
        return row[CHUNK_ID_COLUMN]

    def __setitem__(self, *args):
        raise NotImplementedError(
            "There is no reason for ChunkIdEncoder to be updated now."
        )

    def __getitem__(
        self, local_sample_index: int, return_row_index: bool = False
    ) -> Union[List[ENCODING_DTYPE], Tuple[List[ENCODING_DTYPE], Any]]:
        """Returns a list of chunk IDs. If the sample is not tiled it will always return a tuple of length 1."""

        # TODO: this method can probably be generalized into base class `__getitem__` with an extra parameter

        root_chunk_id, root_chunk_id_index = super().__getitem__(
            local_sample_index, return_row_index=True
        )  # type: ignore
        root_chunk_last_seen_index = self._encoded[
            root_chunk_id_index, LAST_SEEN_INDEX_COLUMN
        ]
        returns = [root_chunk_id]

        # TODO: explain this:
        c = 1
        while True:
            try:
                tile_chunk_id, tile_chunk_last_seen_index = self._encoded[
                    root_chunk_id_index + c
                ]

                if tile_chunk_last_seen_index == root_chunk_last_seen_index:
                    returns.append(tile_chunk_id)
                else:
                    break

            except IndexError:
                break
            c += 1

        if return_row_index:
            # TODO: note this in the docstring
            return returns, root_chunk_id_index

        return returns
