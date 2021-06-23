from typing import Tuple
import numpy as np
from hub.core.typing import StorageProvider
from hub.core.meta.tensor_meta import TensorMeta
from hub import constants


class ChunkEngine:
    def __init__(
        self,
        key: str,
        storage: StorageProvider,
        tensor_meta: TensorMeta,
        min_chunk_size_target=constants.DEFAULT_CHUNK_MIN_TARGET,
        max_chunk_size=constants.DEFAULT_CHUNK_MAX_SIZE,
    ):
        self.key = key
        self.storage = storage
        self.tensor_meta = tensor_meta

        self.min_chunk_size_target = min_chunk_size_target
        self.max_chunk_size = max_chunk_size

        self._chunks = {}

    @property
    def num_chunks(self):
        raise NotImplementedError()

    def extend(self, array: np.ndarray):
        raise NotImplementedError()

    def append(self, array: np.ndarray):
        raise NotImplementedError()

    def get_sample(self, sample_index: int):
        raise NotImplementedError()

    @staticmethod
    def calculate_bytes(shape: Tuple[int], dtype: np.dtype):
        return np.prod(shape) * dtype().itemsize
