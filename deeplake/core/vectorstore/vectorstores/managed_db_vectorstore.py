from abc import ABC, abstractmethod
import logging
import pathlib
from typing import Optional, Any, Iterable, List, Dict, Union, Callable

import numpy as np

from deeplake.core.vector_index.distance_type import DistanceType
from deeplake.util.dataset import try_flushing
from deeplake.util.exceptions import IncorrectEmbeddingShapeError

try:
    from indra import api  # type: ignore

    _INDRA_INSTALLED = True
except Exception:  # pragma: no cover
    _INDRA_INSTALLED = False  # pragma: no cover

import deeplake
from deeplake.constants import (
    DEFAULT_VECTORSTORE_TENSORS,
    VECTORSTORE_EXTEND_MAX_SIZE_BY_HTYPE,
    VECTORSTORE_EXTEND_MAX_SIZE,
)
from deeplake.core.vectorstore import utils
from deeplake.core.vectorstore.vector_search import vector_search
from deeplake.core.vectorstore.vector_search import dataset as dataset_utils
from deeplake.core.vectorstore.vector_search import filter as filter_utils
from deeplake.core.vectorstore.vector_search.indra import index

from deeplake.util.bugout_reporter import (
    feature_report_path,
    deeplake_reporter,
)

logger = logging.getLogger(__name__)


class ManagedDBVectorStore(ABC):
    """Base class for VectorStore"""

    def __init__(
        self,
        path: Union[str, pathlib.Path],
        tensor_params: List[Dict[str, object]] = DEFAULT_VECTORSTORE_TENSORS,
        embedding_function: Optional[Callable] = None,
        read_only: Optional[bool] = None,
        token: Optional[str] = None,
        overwrite: bool = False,
        verbose: bool = True,
        creds: Optional[Union[Dict, str]] = None,
        runtime: Dict = {"tensor_db": True},
        **kwargs: Any,
    ) -> None:
        self.read_only = read_only
        self.runtime = runtime
        self.exec_option = "tensor_db"  # TODO: remove it, need to hadle all edge cases.
        # TODO: add create_or_load request from backend

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
        (
            embedding_function,
            embedding_data,
            embedding_tensor,
            tensors,
        ) = utils.parse_tensors_kwargs(
            tensors, embedding_function, embedding_data, embedding_tensor
        )

        (
            embedding_function,
            embedding_data,
            embedding_tensor,
            tensors,
        ) = utils.parse_add_arguments(
            dataset=self.dataset,
            initial_embedding_function=self.embedding_function,
            embedding_function=embedding_function,
            embedding_data=embedding_data,
            embedding_tensor=embedding_tensor,
            **tensors,
        )

        processed_tensors, id_ = dataset_utils.preprocess_tensors(
            embedding_data, embedding_tensor, self.dataset, **tensors
        )

        assert id_ is not None
        utils.check_length_of_each_tensor(processed_tensors)

        first_item = next(iter(processed_tensors))

        htypes = [
            self.dataset[item].meta.htype for item in self.dataset.tensors
        ]  # Inspect raw htype (not parsed htype like tensor.htype) in order to avoid parsing links and sequences separately.
        threshold_by_htype = [
            VECTORSTORE_EXTEND_MAX_SIZE_BY_HTYPE.get(h, int(1e10)) for h in htypes
        ]
        extend_threshold = min(threshold_by_htype + [VECTORSTORE_EXTEND_MAX_SIZE])

        if len(processed_tensors[first_item]) <= extend_threshold:
            if embedding_function:
                for func, data, tensor in zip(
                    embedding_function, embedding_data, embedding_tensor
                ):
                    embedded_data = func(data)
                    try:
                        embedded_data = np.array(embedded_data, dtype=np.float32)
                    except ValueError:
                        raise IncorrectEmbeddingShapeError()

                    if len(embedded_data) == 0:
                        raise ValueError("embedding function returned empty list")

                    processed_tensors[tensor] = embedded_data
        # TODO: send processed_tensors to the backend

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
        # TODO: Use the default logic
        ...

    def delete(
        self,
        row_ids: Optional[List[str]] = None,
        ids: Optional[List[str]] = None,
        filter: Optional[Union[Dict, Callable]] = None,
        query: Optional[str] = None,
        exec_option: Optional[str] = None,
        delete_all: Optional[bool] = None,
    ) -> bool:
        if not row_ids:
            row_ids = dataset_utils.search_row_ids(
                dataset=self.dataset,
                search_fn=self.search,
                ids=ids,
                filter=filter,
                query=query,
                select_all=delete_all,
                exec_option=exec_option or self.exec_option,
            )

        # TODO: send delete_all and row_ids to backend.
        # delete_all and row_ids can't be used at the same time,
        # both of them can'be specified at the same time.

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
        (
            embedding_function,
            embedding_source_tensor,
            embedding_tensor,
        ) = utils.parse_update_arguments(
            dataset=self.dataset,
            embedding_function=embedding_function,
            initial_embedding_function=self.embedding_function,
            embedding_source_tensor=embedding_source_tensor,
            embedding_tensor=embedding_tensor,
        )

        if not row_ids:
            row_ids = dataset_utils.search_row_ids(
                dataset=self.dataset,
                search_fn=self.search,
                ids=ids,
                filter=filter,
                query=query,
                exec_option=exec_option or self.exec_option,
            )

        embedding_tensor_data = utils.convert_embedding_source_tensor_to_embeddings(
            dataset=self.dataset,
            embedding_source_tensor=embedding_source_tensor,
            embedding_tensor=embedding_tensor,
            embedding_function=embedding_function,
            row_ids=row_ids,
        )
        # TODO: send embedding_tensor_data and row_ids to backend

    @staticmethod
    def delete_by_path(
        path: Union[str, pathlib.Path],
        token: Optional[str] = None,
        creds: Optional[Union[Dict, str]] = None,
    ) -> None:
        # TODO: request to backend to delete the dataset
        ...

    def commit(self, allow_empty: bool = True) -> None:
        # TODO: request to backend to commit the dataset
        ...

    def tensors(self):
        # TODO: request to backend to get tensors
        ...

    def summary(self):
        # TODO: request to backend to get summary
        ...

    def __len__(self):
        # TODO: request to backend to get length
        ...
