from .indexer import Indexer

import numpy as np

from abc import abstractmethod
from typing import List


class MutableIndexer(Indexer):
    @property
    def indexer(self) -> Indexer:
        return self

    @abstractmethod
    def save(self) -> bytes:
        pass

    @abstractmethod
    def add_sample(self, vector: np.ndarray):
        pass

    @abstractmethod
    def remove_sample(self, index: int):
        pass

    @abstractmethod
    def add_samples(self, vectors: List[np.ndarray]):
        pass

    @abstractmethod
    def remove_samples(self, indices: List[int]):
        pass
