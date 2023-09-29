from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from deeplake.core.dataset import Dataset
from deeplake.core.vectorstore.deeplake_vectorstore import VectorStore
from deeplake.core.vectorstore.deep_memory import DeepMemory
from deeplake.constants import DEFAULT_DEEPMEMORY_DISTANCE_METRIC
from deeplake.util.exceptions import LockedException


class DeepMemoryVectorStore(VectorStore):
    def __init__(self, *arg, **kwargs):
        super().__init__(*arg, **kwargs)
        try:
            self.deep_memory = DeepMemory(
                self.dataset,
                token=self.token,
                embedding_function=self.embedding_function,
            )
        except Exception as e:
            if e == LockedException:
                raise e

            self.deep_memory = None

    def search(
        self,
        embedding_data: Optional[Union[str, List[str]]] = None,
        embedding_function: Optional[Callable[..., Any]] = None,
        embedding: Optional[Union[List[float], np.ndarray]] = None,
        k: int = 4,
        distance_metric: Optional[str] = None,
        query: Optional[str] = None,
        filter: Optional[Union[Dict, Callable[..., Any]]] = None,
        exec_option: Optional[str] = None,
        embedding_tensor: str = "embedding",
        return_tensors: Optional[List[str]] = None,
        return_view: bool = False,
        deep_memory=True,
    ) -> Union[Dict, Dataset]:
        if exec_option is not None and exec_option != "tensor_db":
            self.logger.warning(
                "Specifying `exec_option` is not supported for this dataset. "
                "The search will be executed on the Deep Lake Managed Database."
            )

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
            exec_option="tensor_db",
            embedding_tensor=embedding_tensor,
            return_tensors=return_tensors,
            return_view=return_view,
        )
