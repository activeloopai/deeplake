from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from deeplake.core.dataset import Dataset
from deeplake.core.vectorstore.deeplake_vectorstore import VectorStore
from deeplake.core.vectorstore.deep_memory import DeepMemory
from deeplake.constants import DEFAULT_DEEPMEMORY_DISTANCE_METRIC


class DeepMemoryVectorStore(VectorStore):
    def __init__(self, client, *arg, **kwargs):
        super().__init__(*arg, **kwargs)
        self.deep_memory = DeepMemory(
            self.dataset,
            token=self.token,
            embedding_function=self.embedding_function,
            client=client,
            creds=self.creds,
        )

    def search(
        self,
        embedding_data: Union[str, List[str], None] = None,
        embedding_function: Optional[Callable] = None,
        embedding: Optional[Union[List[float], np.ndarray]] = None,
        k: int = 4,
        distance_metric: Optional[str] = None,
        query: Optional[str] = None,
        filter: Optional[Union[Dict, Callable]] = None,
        exec_option: Optional[str] = None,
        embedding_tensor: str = "embedding",
        return_tensors: Optional[List[str]] = None,
        return_view: bool = False,
        deep_memory: bool = False,
    ) -> Union[Dict, Dataset]:
        if deep_memory and not distance_metric:
            distance_metric = DEFAULT_DEEPMEMORY_DISTANCE_METRIC

        return super().search(
            embedding_data=embedding_data,
            embedding_function=embedding_function,
            embedding=embedding,
            k=k,
            distance_metric=distance_metric,
            query=query,
            filter=filter,
            exec_option=exec_option,
            embedding_tensor=embedding_tensor,
            return_tensors=return_tensors,
            return_view=return_view,
            deep_memory=deep_memory,
        )
