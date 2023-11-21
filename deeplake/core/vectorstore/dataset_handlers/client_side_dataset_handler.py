import logging
import pathlib
import pickle
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

import deeplake
from deeplake.client.utils import read_token
from deeplake.constants import (
    DEFAULT_VECTORSTORE_DISTANCE_METRIC,
    _INDEX_OPERATION_MAPPING,
)
from deeplake.core import index_maintenance
from deeplake.core.dataset import Dataset
from deeplake.core.vectorstore import utils
from deeplake.core.vectorstore.dataset_handlers.dataset_handler_base import DHBase
from deeplake.core.vectorstore.deep_memory.deep_memory import (
    use_deep_memory,
    DeepMemory,
)
from deeplake.core.vectorstore.vector_search import dataset as dataset_utils
from deeplake.core.vectorstore.vector_search import vector_search
from deeplake.util.bugout_reporter import feature_report_path
from deeplake.util.exceptions import DeepMemoryWaitingListError


class ComplexInitilization:
    def initialize(
        self,
        client_instance: Any,
    ):
        client_instance.index_params = utils.parse_index_params(
            client_instance.index_params
        )
        kwargs = {"index_params": client_instance.index_params}
        client_instance.dataset = (
            client_instance.dataset
            or dataset_utils.create_or_load_dataset(
                tensor_params=client_instance.tensor_params,
                dataset_path=client_instance.path,
                token=client_instance.token,
                creds=client_instance.creds,
                logger=client_instance.logger,
                read_only=client_instance.read_only,
                exec_option=client_instance._exec_option,
                embedding_function=client_instance.embedding_function,
                overwrite=client_instance.overwrite,
                runtime=client_instance.runtime,
                org_id=client_instance.org_id,
                branch=client_instance.branch,
                **kwargs,
            )
        )
        client_instance.distance_metric_index = (
            index_maintenance.index_operation_vectorstore(
                client_instance,
                dml_type=_INDEX_OPERATION_MAPPING["ADD"],
                rowids=list(range(0, len(client_instance.dataset))),
            )
        )
        self.deep_memory = DeepMemory(
            dataset_or_path=client_instance.path,
            token=client_instance.token,
            logger=client_instance.logger,
            embedding_function=client_instance.embedding_function,
            creds=client_instance.creds,
        )


class LightweightInitilization:
    def initialize(
        self,
        client_instance: Any,
        serialized_vectorstore: Any,
    ):
        client_instance.distance_metric_index = getattr(
            serialized_vectorstore, "distance_metric_index", None
        )
        client_instance.deep_memory = getattr(
            serialized_vectorstore, "deep_memory", None
        )
        client_instance.dataset = getattr(serialized_vectorstore, "dataset", None)


def initialise_dh(client_instance: Any, deserialized_vectorstore: Any):
    if deserialized_vectorstore:
        return LightweightInitilization().initialize(
            client_instance, deserialized_vectorstore
        )
    return ComplexInitilization().initialize(client_instance)


class ClientSideDH(DHBase):
    def __init__(
        self,
        path: Union[str, pathlib.Path],
        dataset: Dataset,
        tensor_params: List[Dict[str, object]],
        embedding_function: Any,
        read_only: bool,
        ingestion_batch_size: int,
        index_params: Dict[str, Union[int, str]],
        num_workers: int,
        exec_option: str,
        token: str,
        overwrite: bool,
        verbose: bool,
        runtime: Dict,
        creds: Union[Dict, str],
        org_id: str,
        logger: logging.Logger,
        branch: str,
        **kwargs: Any,
    ):
        super().__init__(
            path=path,
            dataset=dataset,
            tensor_params=tensor_params,
            embedding_function=embedding_function,
            read_only=read_only,
            ingestion_batch_size=ingestion_batch_size,
            index_params=index_params,
            num_workers=num_workers,
            exec_option=exec_option,
            token=token,
            overwrite=overwrite,
            verbose=verbose,
            runtime=runtime,
            creds=creds,
            org_id=org_id,
            logger=logger,
            branch=branch,
            **kwargs,
        )
        # using strategy pattern to initialize the class
        initialise_dh(self, self.deserialized_vectorstore)

    def add(
        self,
        embedding_function: Union[Callable, List[Callable]],
        embedding_data: Union[List, List[List]],
        embedding_tensor: Union[str, List[str]],
        return_ids: bool,
        rate_limiter: Dict,
        **tensors,
    ):
        feature_report_path(
            path=self.bugout_reporting_path,
            feature_name="vs.add",
            parameters={
                "tensors": list(tensors.keys()) if tensors else None,
                "embedding_tensor": embedding_tensor,
                "return_ids": return_ids,
                "embedding_function": True if embedding_function is not None else False,
                "embedding_data": True if embedding_data is not None else False,
            },
            token=self.token,
            username=self.username,
        )
        (
            embedding_function,
            embedding_data,
            embedding_tensor,
            tensors,
        ) = utils.parse_tensors_kwargs(
            tensors,
            embedding_function,
            embedding_data,
            embedding_tensor,
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

        dataset_utils.extend_or_ingest_dataset(
            processed_tensors=processed_tensors,
            dataset=self.dataset,
            embedding_function=embedding_function,
            embedding_data=embedding_data,
            embedding_tensor=embedding_tensor,
            rate_limiter=rate_limiter,
            logger=self.logger,
        )

        if self.verbose:
            self.dataset.summary()

        if return_ids:
            return id_
        return None

    @use_deep_memory
    def search(
        self,
        embedding_data: Union[str, List[str]],
        embedding_function: Callable,
        embedding: Union[List[float], np.ndarray],
        k: int,
        distance_metric: str,
        query: str,
        filter: Union[Dict, Callable],
        exec_option: str,
        embedding_tensor: str,
        return_tensors: List[str],
        return_view: bool,
        deep_memory: bool,
    ) -> Union[Dict, Dataset]:
        feature_report_path(
            path=self.bugout_reporting_path,
            feature_name="vs.search",
            parameters={
                "embedding_data": True if embedding_data is not None else False,
                "embedding_function": True if embedding_function is not None else False,
                "k": k,
                "distance_metric": distance_metric,
                "query": query[0:100] if query is not None else False,
                "filter": True if filter is not None else False,
                "exec_option": exec_option,
                "embedding_tensor": embedding_tensor,
                "embedding": True if embedding is not None else False,
                "return_tensors": return_tensors,
                "return_view": return_view,
            },
            token=self.token,
            username=self.username,
        )

        if exec_option is None and self.exec_option != "python" and callable(filter):
            self.logger.warning(
                'Switching exec_option to "python" (runs on client) because filter is specified as a function. '
                f'To continue using the original exec_option "{self.exec_option}", please specify the filter as a dictionary or use the "query" parameter to specify a TQL query.'
            )
            exec_option = "python"

        exec_option = exec_option or self.exec_option

        if deep_memory and not self.deep_memory:
            raise DeepMemoryWaitingListError()

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
            self.dataset, return_tensors, embedding_tensor, return_view
        )
        embedding_function = utils.create_embedding_function(embedding_function)
        query_emb: Optional[Union[List[float], np.ndarray[Any, Any]]] = None
        if query is None:
            query_emb = dataset_utils.get_embedding(
                embedding,
                embedding_data,
                embedding_function=embedding_function or self.embedding_function,
            )

        if self.distance_metric_index:
            distance_metric = index_maintenance.parse_index_distance_metric_from_params(
                self.logger, self.distance_metric_index, distance_metric
            )

        distance_metric = distance_metric or DEFAULT_VECTORSTORE_DISTANCE_METRIC

        return vector_search.search(
            query=query,
            logger=self.logger,
            filter=filter,
            query_embedding=query_emb,
            k=k,
            distance_metric=distance_metric,
            exec_option=exec_option,
            deeplake_dataset=self.dataset,
            embedding_tensor=embedding_tensor,
            return_tensors=return_tensors,
            return_view=return_view,
            deep_memory=deep_memory,
            token=self.token,
            org_id=self.org_id,
        )

    def delete(
        self,
        row_ids: List[int],
        ids: List[str],
        filter: Union[Dict, Callable],
        query: str,
        exec_option: str,
        delete_all: bool,
    ) -> bool:
        feature_report_path(
            path=self.bugout_reporting_path,
            feature_name="vs.delete",
            parameters={
                "ids": True if ids is not None else False,
                "row_ids": True if row_ids is not None else False,
                "query": query[0:100] if query is not None else False,
                "filter": True if filter is not None else False,
                "exec_option": exec_option,
                "delete_all": delete_all,
            },
            token=self.token,
            username=self.username,
        )

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

        (
            self.dataset,
            dataset_deleted,
        ) = dataset_utils.delete_all_samples_if_specified(
            self.dataset,
            delete_all,
        )
        if dataset_deleted:
            index_maintenance.index_operation_vectorstore(
                self,
                dml_type=_INDEX_OPERATION_MAPPING["REMOVE"],
                rowids=row_ids,
                index_delete=True,
            )
            return True

        self.dataset.pop_multiple(row_ids)

        return True

    def update_embedding(
        self,
        row_ids: List[str],
        ids: List[str],
        filter: Union[Dict, Callable],
        query: str,
        exec_option: str,
        embedding_function: Union[Callable, List[Callable]],
        embedding_source_tensor: Union[str, List[str]],
        embedding_tensor: Union[str, List[str]],
    ):
        feature_report_path(
            path=self.bugout_reporting_path,
            feature_name="vs.delete",
            parameters={
                "ids": True if ids is not None else False,
                "row_ids": True if row_ids is not None else False,
                "query": query[0:100] if query is not None else False,
                "filter": True if filter is not None else False,
                "exec_option": exec_option,
            },
            token=self.token,
            username=self.username,
        )

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

        self.dataset[row_ids].update(embedding_tensor_data)

    def commit(self, allow_empty: bool = True) -> None:
        """Commits the Vector Store.

        Args:
            allow_empty (bool): Whether to allow empty commits. Defaults to True.
        """
        self.dataset.commit(allow_empty=allow_empty)

    def checkout(self, branch: str = "main") -> None:
        """Checkout the Vector Store to a specific branch.

        Args:
            branch (str): Branch name to checkout. Defaults to "main".
        """
        self.dataset.checkout(branch)

    def tensors(self):
        """Returns the list of tensors present in the dataset"""
        return self.dataset.tensors

    def summary(self):
        """Prints a summary of the dataset"""
        return self.dataset.summary()

    def __len__(self):
        """Length of the dataset"""
        return len(self.dataset)
