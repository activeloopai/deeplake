from .distance_type import DistanceType

import numpy as np

from abc import ABC, abstractmethod
from typing import Union, List, Tuple, Dict, Any


class Indexer(ABC):
    @abstractmethod
    def load(self, data: bytes):
        pass

    @property
    @abstractmethod
    def distance_type(self) -> DistanceType:
        pass

    @property
    @abstractmethod
    def params(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def search_top_k(
        self, vector: np.ndarray, k: int, return_score: bool
    ) -> Union[List[int], List[Tuple[int, float]]]:
        pass
