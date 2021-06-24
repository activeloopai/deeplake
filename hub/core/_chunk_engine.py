from hub.util.keys import get_chunk_index_meta_key, get_chunk_key
from hub.core.chunk import Chunk
from math import ceil
from hub.core.meta.index_meta import IndexMeta
from hub.core.meta.encode.chunk_name import ChunkNameEncoder
from hub.core.index.index import Index
from typing import Dict, List, Tuple
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
        return self._chunk_names_encoder.num_chunks

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
            last_chunk_size = last_chunk.num_data_bytes
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

                # can be 0, in that case it is only partial
                full_samples_added_to_chunk = len(chunk_bytes) // bytes_per_sample

                # cannot be 0
                full_and_partial_samples_added_to_chunk = ceil(
                    len(chunk_bytes) / bytes_per_sample
                )

                # TODO: this code can probably be moved into Chunk
                last_chunk._shape_encoder.add_shape(
                    sample_shape, full_and_partial_samples_added_to_chunk
                )
                last_chunk._byte_positions_encoder.add_byte_position(
                    bytes_per_sample, full_and_partial_samples_added_to_chunk
                )
                chunk_name = self._chunk_names_encoder.extend_chunk(
                    full_samples_added_to_chunk, connected_to_next=connected_to_next
                )

                chunk_key = get_chunk_key(self.key, chunk_name)
                self.storage[chunk_key] = last_chunk

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

            # can be 0, in that case it is only partial
            full_samples_added_to_chunk = len(chunk_bytes) // bytes_per_sample

            # cannot be 0
            full_and_partial_samples_added_to_chunk = ceil(
                len(chunk_bytes) / bytes_per_sample
            )

            new_chunk._shape_encoder.add_shape(
                sample_shape, full_and_partial_samples_added_to_chunk
            )
            new_chunk._byte_positions_encoder.add_byte_position(
                bytes_per_sample, full_and_partial_samples_added_to_chunk
            )
            chunk_name = self._chunk_names_encoder.append_chunk(
                full_samples_added_to_chunk, connected_to_next=connected_to_next
            )

            # TODO: somehow extract name from self._chunk_names_encoder and assign / update `Chunk` instances with it

            chunk_key = get_chunk_key(self.key, chunk_name)
            self.storage[chunk_key] = new_chunk

        self.tensor_meta.update_with_sample(sample_shape, sample_dtype)
        self.tensor_meta.length += sample_count

        self.storage[
            get_chunk_index_meta_key(self.key)
        ] = self._chunk_names_encoder.tobytes()

    def append(self, array: np.ndarray):
        self.extend(np.expand_dims(array, 0))

    def get_samples(self, index: Index, aslist: bool = False):
        # TODO: implement this!

        # TODO: maybe this can be more efficient?
        samples = []

        # TODO: indexing here doesn't work...
        # for global_sample_index in index.values[0].indices(self.num_samples):
        for global_sample_index in range(self.num_samples):
            chunk_names = self._chunk_names_encoder.get_chunk_names(global_sample_index)
            sample_bytes = bytearray()

            for chunk_name in chunk_names:
                chunk_key = get_chunk_key(self.key, chunk_name)
                chunk = self.storage[chunk_key]

                local_sample_index = global_sample_index  # TODO
                sample_bytes += chunk.get_sample_bytes(local_sample_index)
                sample_shape = chunk.get_sample_shape(local_sample_index)

            a = np.frombuffer(sample_bytes, dtype=self.tensor_meta.dtype).reshape(
                sample_shape
            )

            samples.append(a)

        if aslist:
            return samples

        # TODO: if dynamic array catch this error early
        return np.array(samples)

    def get_last_chunk(self) -> Chunk:
        if self.num_chunks == 0:
            return None

        chunk_name = self._chunk_names_encoder.get_name_for_chunk(-1)
        chunk_key = get_chunk_key(self.key, chunk_name)

        chunk = self.storage[chunk_key]
        if type(chunk) == bytes:
            return Chunk.frombuffer(chunk)
        return chunk

    @staticmethod
    def calculate_bytes(shape: Tuple[int], dtype: np.dtype):
        return np.prod(shape) * dtype().itemsize

    def min_chunks_required_for_num_bytes(self, num_bytes: int) -> int:
        """Calculates the minimum number of chunks in which data of given size can be fit."""
        return ceil(num_bytes / self.max_chunk_size)
