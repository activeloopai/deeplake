from .distance_type import DistanceType

import numpy as np

from abc import ABC, abstractmethod
from typing import Union, List, Tuple, Dict, Any


class Indexer(ABC):
    @abstractmethod
    def search_knn(
        self, vector: np.ndarray, k: int, return_score: bool
    ) -> Union[List[int], List[Tuple[int, float]]]:
        pass
