from hub.core.meta.index_meta import IndexMeta
from hub.core.index.index import Index
from typing import Tuple
import numpy as np
from hub.core.typing import StorageProvider
from hub.core.meta.tensor_meta import TensorMeta
from hub import constants

import hub.core.tensor as tensor


class ChunkEngine:
    def __init__(
        self,
        key: str,
        storage: StorageProvider,
        min_chunk_size_target=constants.DEFAULT_CHUNK_MIN_TARGET,
        max_chunk_size=constants.DEFAULT_CHUNK_MAX_SIZE,
    ):
        self.key = key
        self.storage = storage

        self.min_chunk_size_target = min_chunk_size_target
        self.max_chunk_size = max_chunk_size

        self._chunks = {}

        # TODO: remove this!!!!!
        tensor.create_tensor(self.key, self.storage)
        self.tensor_meta = TensorMeta.load(self.key, self.storage)
        self.index_meta = IndexMeta.load(self.key, self.storage)

    @property
    def num_chunks(self):
        # TODO: implement this!
        raise NotImplementedError()

    @property
    def num_samples(self):
        # TODO: implement this!
        return self.tensor_meta.length

    def extend(self, array: np.ndarray):
        # TODO: implement this!
        tensor.extend_tensor(
            array, self.key, self.storage, self.tensor_meta, self.index_meta
        )

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

    @staticmethod
    def calculate_bytes(shape: Tuple[int], dtype: np.dtype):
        return np.prod(shape) * dtype().itemsize
