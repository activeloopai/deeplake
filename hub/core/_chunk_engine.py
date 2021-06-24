from hub.core.chunk import Chunk
from math import ceil
from hub.core.meta.index_meta import IndexMeta
from hub.core.meta.encode.chunk_name import ChunkNameEncoder
from hub.core.index.index import Index
from typing import List, Tuple
import numpy as np
from hub.core.typing import StorageProvider
from hub.core.meta.tensor_meta import TensorMeta
from hub import constants
from hub.core.chunk_engine.flatten import row_wise_to_bytes

import hub.core.tensor as tensor


def _should_chunk_be_connected_to_next(
    chunk: Chunk,
    bytes_written_so_far: int,
    bytes_per_sample: int,
    bytes_written_to_chunk: int,
) -> bool:
    # can only say this chunk is connected to next if it is full
    if chunk.has_space:
        return False
    return (bytes_written_so_far + bytes_written_to_chunk) % bytes_per_sample == 0


class ChunkEngine:
    def __init__(
        self,
        key: str,
        storage: StorageProvider,
        min_chunk_size_target: int = constants.DEFAULT_CHUNK_MIN_TARGET,
        max_chunk_size: int = constants.DEFAULT_CHUNK_MAX_SIZE,
        create_tensor: bool = False,
    ):
        self.key = key
        self.storage = storage
        self.cached_chunks: List[Chunk] = []

        self.min_chunk_size_target = min_chunk_size_target
        self.max_chunk_size = max_chunk_size

        # TODO: remove this block!!!!!
        if create_tensor:
            tensor.create_tensor(self.key, self.storage)

        self.tensor_meta = TensorMeta.load(self.key, self.storage)
        self.index_meta = IndexMeta.load(self.key, self.storage)

        # TODO: load if already exists
        self._chunk_names_encoder = ChunkNameEncoder()

    @property
    def num_chunks(self):
        # TODO: implement this!
        raise NotImplementedError()

    @property
    def num_samples(self):
        # TODO: implement this!
        return self.tensor_meta.length

    def extend(self, array: np.ndarray):
        sample_count = array.shape[0]
        sample_shape = array.shape[1:]
        sample_dtype = array.dtype
        bytes_per_sample = array.nbytes // sample_count

        self.tensor_meta.check_compatibility(sample_shape, sample_dtype)

        # TODO: replace with space filling curves to optimize access patterns
        # TODO: space filling curves will break the logic below because row_wise_to_bytes is C ordered
        data_buffer = row_wise_to_bytes(array)

        last_chunk = self.get_last_chunk()
        extend_previous = last_chunk is not None and last_chunk.has_space
        bytes_written_so_far = 0

        if extend_previous:
            last_chunk_size = len(last_chunk)
            min_chunks_required = self.min_chunks_required_for_num_bytes(
                len(data_buffer)
            )

            extra_bytes = min(len(data_buffer), self.max_chunk_size - last_chunk_size)
            combined_min_chunks_required = self.min_chunks_required_for_num_bytes(
                len(data_buffer) + last_chunk_size
            )

            # combine if count is same
            if combined_min_chunks_required == min_chunks_required:
                # TODO: update `last_chunk`'s shape / byte positions encoding
                chunk_bytes = data_buffer[0:extra_bytes]

                last_chunk.extend(chunk_bytes)
                connected_to_next = _should_chunk_be_connected_to_next(
                    last_chunk, bytes_written_so_far, bytes_per_sample, len(chunk_bytes)
                )

                data_buffer = data_buffer[extra_bytes:]
                bytes_written_so_far += len(chunk_bytes)
                samples_added_to_chunk = max(1, len(chunk_bytes) // bytes_per_sample)

                last_chunk._shape_encoder.add_shape(sample_shape, sample_count)
                self._chunk_names_encoder.extend_chunk(
                    samples_added_to_chunk, connected_to_next=connected_to_next
                )

        # each iteration of this loop will create a new chunk
        while len(data_buffer) > 0:
            new_chunk = Chunk(self.min_chunk_size_target, self.max_chunk_size)
            end_byte = min(len(data_buffer), self.max_chunk_size)
            chunk_bytes = data_buffer[:end_byte]

            # chunk_content = content[:end_byte]  # type: ignore
            # _write_chunk(chunk_content, storage, chunk_names, key)

            """ inside _write_chunk:
            chunk_name = chunk_name or _generate_chunk_name()
            chunk_names.append(chunk_name)
            chunk_key = get_chunk_key(key, chunk_name)
            storage[chunk_key] = content
            """

            # TODO: update `new_chunk`'s shape / byte positions encoding

            new_chunk.extend(chunk_bytes)
            connected_to_next = _should_chunk_be_connected_to_next(
                new_chunk, bytes_written_so_far, bytes_per_sample, len(chunk_bytes)
            )

            data_buffer = data_buffer[end_byte:]
            bytes_written_so_far += len(chunk_bytes)
            samples_added_to_chunk = max(1, len(chunk_bytes) // bytes_per_sample)

            new_chunk._shape_encoder.add_shape(sample_shape, samples_added_to_chunk)
            self._chunk_names_encoder.append_chunk(
                samples_added_to_chunk, connected_to_next=connected_to_next
            )

            # TODO: turn bool arg into 2 methods instead of 1
            # TODO: somehow extract name from self._chunk_names_encoder and assign / update `Chunk` instances with it
            # self._chunk_names_encoder.add_samples_to_chunk(
            # sample_count, extend_previous
            # )

            # tensor.extend_tensor(
            # array, self.key, self.storage, self.tensor_meta, self.index_meta
            # )

            self.cached_chunks.append(new_chunk)

        print(bytes_per_sample)
        print(self.cached_chunks)
        print(self.cached_chunks[0]._shape_encoder.num_samples)

        self.tensor_meta.length += sample_count
        self.tensor_meta.update_with_sample(sample_shape, sample_dtype)

        """
        if _chunk_has_space(last_chunk, tensor_meta.chunk_size):
            last_chunk_size = len(last_chunk)
            chunk_ct_content = _min_chunk_ct_for_data_size(len(content))

            extra_bytes = min(len(content), DEFAULT_CHUNK_MAX_SIZE - last_chunk_size)
            combined_chunk_ct = _min_chunk_ct_for_data_size(len(content) + last_chunk_size)

            if combined_chunk_ct == chunk_ct_content:  # combine if count is same
                start_byte = index_meta.entries[-1]["end_byte"]
                end_byte = start_byte + extra_bytes

                chunk_content = bytearray(last_chunk) + content[0:extra_bytes]
                _write_chunk(chunk_content, storage, chunk_names, key, last_chunk_name)

                content = content[extra_bytes:]

        while len(content) > 0:
            end_byte = min(len(content), DEFAULT_CHUNK_MAX_SIZE)

            chunk_content = content[:end_byte]  # type: ignore
            _write_chunk(chunk_content, storage, chunk_names, key)

            content = content[end_byte:]

            # TODO: turn bool arg into 2 methods instead of 1
            # TODO: somehow extract name from self._chunk_names_encoder and assign / update `Chunk` instances with it
            self._chunk_names_encoder.add_samples_to_chunk(sample_count, extend_previous)

            # tensor.extend_tensor(
            # array, self.key, self.storage, self.tensor_meta, self.index_meta
            # )
        """

    def append(self, array: np.ndarray):
        # TODO: implement this!
        tensor.append_tensor(
            array, self.key, self.storage, self.tensor_meta, self.index_meta
        )

    def get_sample(self, index: Index, aslist: bool = False):
        # TODO: implement this!
        return tensor.read_samples_from_tensor(
            self.key, self.storage, index, aslist=aslist
        )

    def get_last_chunk(self) -> Chunk:
        if len(self.cached_chunks) == 0:
            # TODO: read from storage
            return None

        return self.cached_chunks[-1]

    @staticmethod
    def calculate_bytes(shape: Tuple[int], dtype: np.dtype):
        return np.prod(shape) * dtype().itemsize

    def min_chunks_required_for_num_bytes(self, num_bytes: int) -> int:
        """Calculates the minimum number of chunks in which data of given size can be fit."""
        return ceil(num_bytes / self.max_chunk_size)
