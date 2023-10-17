import logging
import pathlib
from typing import Optional, Any, Iterable, List, Dict, Union, Callable

import numpy as np


import deeplake
from deeplake.constants import (
    DEFAULT_VECTORSTORE_TENSORS,
    VECTORSTORE_EXTEND_MAX_SIZE_BY_HTYPE,
    VECTORSTORE_EXTEND_MAX_SIZE,
)
from deeplake.util.exceptions import IncorrectEmbeddingShapeError
from deeplake.client.managed_client import ManagedServiceClient

from deeplake.core.vectorstore import utils
from deeplake.core.vectorstore.vectorstores.vectorstore_base import VectorStoreBase
from deeplake.core.vectorstore.vector_search import vector_search
from deeplake.core.vectorstore.vector_search import dataset as dataset_utils
from deeplake.core.vectorstore.vector_search import filter as filter_utils

from deeplake.util.bugout_reporter import (
    feature_report_path,
    deeplake_reporter,
)

logger = logging.getLogger(__name__)


class ManagedDBVectorStore:
    def __init__(
        self,
        path: Union[str, pathlib.Path],
        tensor_params: List[Dict[str, object]] = DEFAULT_VECTORSTORE_TENSORS,
        embedding_function: Optional[Callable] = None,
        read_only: Optional[bool] = None,
        token: Optional[str] = None,
        overwrite: bool = False,
        verbose: bool = True,
        **kwargs: Any,
    ) -> None:
        self._path = path
        self.exec_option = "tensor_db"
        self.read_only = read_only
        self.client = ManagedServiceClient(token=token)
        summary = self.client.init_vectorstore(
            path=path,
            overwrite=overwrite,
            tensor_params=tensor_params,
        )

        if verbose:
            print(summary)

        self.embedding_function = embedding_function

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
        dataset_tensors = self.client.get_vectorstore_tensors(self._path)

        (
            embedding_function,
            embedding_data,
            embedding_tensor,
            tensors,
        ) = utils.parse_add_arguments(
            dataset_tensors=dataset_tensors,
            initial_embedding_function=self.embedding_function,
            embedding_function=embedding_function,
            embedding_data=embedding_data,
            embedding_tensor=embedding_tensor,
            **tensors,
        )

        processed_tensors, id_ = dataset_utils.preprocess_tensors(
            embedding_data, embedding_tensor, dataset_tensors, **tensors
        )

        assert id_ is not None
        utils.check_length_of_each_tensor(processed_tensors)

        first_item = next(iter(processed_tensors))

        htypes = [
            dataset_tensor["htype"] for dataset_tensor in dataset_tensors
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

                    if isinstance(embedded_data, np.ndarray):
                        embedded_data = embedded_data.tolist()

                    processed_tensors[tensor] = embedded_data.tolist()

        indices = self.client.extend_vectorstore(self._path, processed_tensors)
        if return_ids:
            return indices

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
        dataset_tensors = self.client.get_vectorstore_tensors(self._path)

        # TODO: Use the default logic
        if exec_option is None and self.exec_option != "python" and callable(filter):
            logger.warning(
                'Switching exec_option to "python" (runs on client) because filter is specified as a function. '
                f'To continue using the original exec_option "{self.exec_option}", please specify the filter as a dictionary or use the "query" parameter to specify a TQL query.'
            )
            exec_option = "python"

        exec_option = exec_option or self.exec_option

        utils.parse_search_args(
            embedding_data=embedding_data,
            embedding_function=embedding_function,
            initial_embedding_function=self.embedding_function,
            embedding=embedding,
            k=k,
            distance_metric=distance_metric,
            query=query,
            filter=filter,
            exec_option=exec_option,
            embedding_tensor=embedding_tensor,
            return_tensors=return_tensors,
        )

        return_tensors = utils.parse_return_tensors(
            dataset_tensors, return_tensors, embedding_tensor, return_view
        )

        query_emb: Optional[Union[List[float], np.ndarray[Any, Any]]] = None
        if query is None:
            query_emb = dataset_utils.get_embedding(
                embedding,
                embedding_data,
                embedding_function=embedding_function or self.embedding_function,
            )
            if isinstance(query_emb, np.ndarray):
                assert (
                    query_emb.ndim == 1 or query_emb.shape[0] == 1
                ), "Query embedding must be 1-dimensional. Please consider using another embedding function for converting query string to embedding."

        if callable(filter):
            raise ValueError(
                "Running filter UDFs with ManagedDBVectorStore is not supported."
            )

        if return_view:
            raise ValueError(
                "Returning Views with ManagedDBVectorStore is not supported."
            )

        tql_string = vector_search.search(
            query=query,
            filter=filter,
            query_embedding=query_emb,
            k=k,
            distance_metric=distance_metric,
            embedding_tensor=embedding_tensor,
            return_tensors=return_tensors,
            return_tql=True,
        )

        data = self.client.search_vectorstore(
            path=self._path,
            query=tql_string,
        )
        return data

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
            row_ids, _ = dataset_utils.search_row_ids(
                dataset=None,
                search_fn=self.search,
                ids=ids,
                filter=filter,
                query=query,
                select_all=delete_all,
                exec_option=exec_option or self.exec_option,
                return_view=False,
            )

        # TODO: send delete_all and row_ids to backend.
        # delete_all and row_ids can't be used at the same time,
        # both of them can'be specified at the same time.
        self.client.remove_vectorstore_indices(self._path, row_ids)

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
        dataset_tensors = self.client.get_vectorstore_tensors(self._path)

        (
            embedding_function,
            embedding_source_tensor,
            embedding_tensor,
        ) = utils.parse_update_arguments(
            dataset=dataset_tensors,
            embedding_function=embedding_function,
            initial_embedding_function=self.embedding_function,
            embedding_source_tensor=embedding_source_tensor,
            embedding_tensor=embedding_tensor,
        )

        if not row_ids:
            row_ids, update_view = dataset_utils.search_row_ids(
                dataset=None,
                search_fn=self.search,
                ids=ids,
                filter=filter,
                query=query,
                exec_option=exec_option or self.exec_option,
            )

            embedding_tensor_data = {}
            if isinstance(embedding_source_tensor, list):
                for (
                    embedding_source_tensor_i,
                    embedding_tensor_i,
                    embedding_fn_i,
                ) in zip(embedding_source_tensor, embedding_tensor, embedding_function):
                    embedding_data = update_view[embedding_source_tensor_i]
                    embedding_tensor_data[embedding_tensor_i] = embedding_fn_i(
                        embedding_data
                    )
                    embedding_tensor_data[embedding_tensor_i] = np.array(
                        embedding_tensor_data[embedding_tensor_i], dtype=np.float32
                    )
            else:
                embedding_data = update_view[embedding_source_tensor_i]
                embedding_tensor_data[embedding_tensor] = embedding_function(
                    embedding_data
                )
                embedding_tensor_data[embedding_tensor] = np.array(
                    embedding_tensor_data[embedding_tensor], dtype=np.float32
                )
        # TODO: send row_ids and embedding_tensor_data to backend.
        return self.client.update_vectorstore_indices(
            self._path,
            row_ids,
            embedding_tensor_data,
        )

    def commit(self, allow_empty: bool = True) -> None:
        # TODO: request to backend to commit the dataset
        pass

    def tensors(self):
        # TODO: request to backend to get tensors
        return self.client.get_vectorstore_tensors(self._path)

    def summary(self):
        # TODO: request to backend to get summary
        return self.client.get_vectorstore_summary(self._path)

    def __len__(self):
        # TODO: request to backend to get length
        return self.client.get_vectorstore_len(self._path)
