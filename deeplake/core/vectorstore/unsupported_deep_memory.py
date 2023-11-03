from deeplake.util.exceptions import DeepMemoryWaitingListError

from typing import List, Tuple, Optional, Callable, Union, Dict, Any

import numpy as np


class DeepMemory:
    """This the class that raises exceptions for users that don't have access to Deep Memory"""

    def __init__(*args, **kwargs):
        # Just some initialization to make sure that the class is not empty
        pass

    def train(
        self,
        queries: List[str],
        relevance: List[List[Tuple[str, int]]],
        embedding_function: Optional[Callable[[str], np.ndarray]] = None,
        token: Optional[str] = None,
    ) -> str:
        raise DeepMemoryWaitingListError()

    def status(self, job_id: str):
        raise DeepMemoryWaitingListError()

    def list_jobs(self, debug=False):
        raise DeepMemoryWaitingListError()

    def evaluate(
        self,
        relevance: List[List[Tuple[str, int]]],
        queries: List[str],
        embedding_function: Optional[Callable[..., List[np.ndarray]]] = None,
        embedding: Optional[Union[List[np.ndarray], List[List[float]]]] = None,
        top_k: List[int] = [1, 3, 5, 10, 50, 100],
        qvs_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, float]]:
        raise DeepMemoryWaitingListError()
