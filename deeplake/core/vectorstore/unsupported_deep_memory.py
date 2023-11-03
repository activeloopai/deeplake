from typing import List, Tuple, Optional, Callable, Union, Dict, Any

import numpy as np


class DeepMemory:
    def __init__(*args, **kwargs):
        pass
    
    def train(
        self,
        queries: List[str],
        relevance: List[List[Tuple[str, int]]],
        embedding_function: Optional[Callable[[str], np.ndarray]] = None,
        token: Optional[str] = None,
    ) -> str:
        raise Exception(
                "Deep Memory is available only for waiting list users. "
                "Please, follow the link and join the waiting list: https://www.deeplake.ai/deepmemory"
            )
        
    def status(self, job_id: str):
        raise Exception(
            "Deep Memory is available only for waiting list users. "
            "Please, follow the link and join the waiting list: https://www.deeplake.ai/deepmemory"
        )
        
    def list_jobs(self, debug=False):
        raise Exception(
            "Deep Memory is available only for waiting list users. "
            "Please, follow the link and join the waiting list: https://www.deeplake.ai/deepmemory"
        )
    
    def evaluate(
        self,
        relevance: List[List[Tuple[str, int]]],
        queries: List[str],
        embedding_function: Optional[Callable[..., List[np.ndarray]]] = None,
        embedding: Optional[Union[List[np.ndarray], List[List[float]]]] = None,
        top_k: List[int] = [1, 3, 5, 10, 50, 100],
        qvs_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, float]]:
        raise Exception(
            "Deep Memory is available only for waiting list users. "
            "Please, follow the link and join the waiting list: https://www.deeplake.ai/deepmemory"
        )   
