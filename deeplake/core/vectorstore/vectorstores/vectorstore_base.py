from abc import ABC, abstractmethod
import logging
import pathlib
from typing import Optional, Any, Iterable, List, Dict, Union, Callable

import numpy as np

from deeplake.util.dataset import try_flushing

try:
    from indra import api  # type: ignore

    _INDRA_INSTALLED = True
except Exception:  # pragma: no cover
    _INDRA_INSTALLED = False  # pragma: no cover

import deeplake
from deeplake.constants import (
    DEFAULT_VECTORSTORE_TENSORS,
)
from deeplake.core.vectorstore import utils
from deeplake.core.vectorstore.vector_search import vector_search
from deeplake.core.vectorstore.vector_search import dataset as dataset_utils
from deeplake.core.vectorstore.vector_search import filter as filter_utils

from deeplake.util.bugout_reporter import (
    feature_report_path,
    deeplake_reporter,
)

logger = logging.getLogger(__name__)


class VectorStoreBase(ABC):
    """Base class for VectorStore"""

    def __init__(
        self,
        path: Union[str, pathlib.Path],
        tensor_params: List[Dict[str, object]] = DEFAULT_VECTORSTORE_TENSORS,
        embedding_function: Optional[Callable] = None,
        read_only: Optional[bool] = None,
        num_workers: int = 0,
        exec_option: str = "auto",
        token: Optional[str] = None,
        overwrite: bool = False,
        verbose: bool = True,
        creds: Optional[Union[Dict, str]] = None,
        **kwargs: Any,
    ) -> None:
        ...

    @abstractmethod
    def add(
        self,
        embedding_function: Optional[Union[Callable, List[Callable]]] = None,
        embedding_data: Optional[Union[List, List[List]]] = None,
        embedding_tensor: Optional[Union[str, List[str]]] = None,
        total_samples_processed: int = 0,
        return_ids: bool = False,
        num_workers: Optional[int] = None,
        **tensors,
    ) -> Optional[List[str]]:
        ...

    @abstractmethod
    def search(
        self,
        embedding_data=None,
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
    ) -> Union[Dict, deeplake.core.dataset.Dataset]:
        ...

    @abstractmethod
    def delete(
        self,
        row_ids: Optional[List[str]] = None,
        ids: Optional[List[str]] = None,
        filter: Optional[Union[Dict, Callable]] = None,
        query: Optional[str] = None,
        exec_option: Optional[str] = None,
        delete_all: Optional[bool] = None,
    ) -> bool:
        ...

    @abstractmethod
    def update_embedding(
        self,
        row_ids: Optional[List[str]] = None,
        ids: Optional[List[str]] = None,
        filter: Optional[Union[Dict, Callable]] = None,
        query: Optional[str] = None,
        exec_option: Optional[str] = None,
        embedding_function: Optional[Union[Callable, List[Callable]]] = None,
        embedding_source_tensor: Union[str, List[str]] = "text",
        embedding_tensor: Optional[Union[str, List[str]]] = None,
    ):
        ...

    @staticmethod
    def delete_by_path(
        path: Union[str, pathlib.Path],
        token: Optional[str] = None,
        creds: Optional[Union[Dict, str]] = None,
    ) -> None:
        ...

    @abstractmethod
    def commit(self, allow_empty: bool = True) -> None:
        ...

    @abstractmethod
    def tensors(self):
        ...

    @abstractmethod
    def summary(self):
        ...

    @abstractmethod
    def __len__(self):
        ...
