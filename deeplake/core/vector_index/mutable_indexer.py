from .indexer import Indexer

import numpy as np

from abc import abstractmethod
from typing import List


class MutableIndexer(Indexer):
    @abstractmethod
    def add_sample(self, vector: np.ndarray, sample_index: int):
        pass

    @abstractmethod
    def remove_sample(self, sample_index: int):
        pass

    @abstractmethod
    def add_samples(self, vectors: List[np.ndarray], sample_indices: List[int]):
        pass

    @abstractmethod
    def remove_samples(self, sample_indices: List[int]):
        pass
