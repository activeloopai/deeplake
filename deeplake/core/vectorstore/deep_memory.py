from typing import Optional, List, Dict, Union, Callable

import numpy as np

from deeplake.core.dataset import Dataset


def deep_memory_available() -> bool:
    # some check whether deepmemory is available
    return True


class DeepMemory:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def train(
        self,
        queries: List[str], 
		embeddings: Union[List[float], np.ndarray, Callable],
		relevance: List[Dict[str, int]],
    ):
        pass

    def cancel(self):
        pass

    def status(self):
        pass

    def list_jobs(self):
        pass

    def evaluate(
        self,
        queries: List[str], 
		embeddings: Union[List[float], np.ndarray, Callable],
		relevance: List[Dict[str, int]],
        evaluate_locally: bool = False,
    ):
        

    def search(self):
        pass


def get_deep_memory() -> Optional[DeepMemory]:
    if deep_memory_available():
        return DeepMemory()
    return None
