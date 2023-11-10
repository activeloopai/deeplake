import logging
import pathlib
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from deeplake.client.managed.managed_client import ManagedServiceClient
from deeplake.client.utils import read_token
from deeplake.constants import MAX_BYTES_PER_MINUTE, TARGET_BYTE_SIZE
from deeplake.core.dataset import Dataset
from deeplake.core.vectorstore.dataset_handlers.dataset_handler_base import DHBase
from deeplake.core.vectorstore.deep_memory.deep_memory import (
    DeepMemory,
    use_deep_memory,
)
from deeplake.core.vectorstore import utils
from deeplake.util.bugout_reporter import feature_report_path
from deeplake.util.path import convert_pathlib_to_string_if_needed, get_path_type


class ManagedSideDH(DHBase):
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
        if embedding_function is not None:
            raise NotImplementedError(
                "ManagedVectorStore does not support embedding_function for now."
            )

        if get_path_type(self.path) != "hub":
            raise ValueError(
                "ManagedVectorStore can only be initialized with a Deep Lake Cloud path."
            )

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
            verbose=True,
            runtime=runtime,
            credscreds=creds,
            org_id=org_id,
            logger=logger,
            **kwargs,
        )

        self.client = ManagedServiceClient(token=self.token)
        self.client.init_vectorstore(
            path=self.path,
            overwrite=overwrite,
            tensor_params=self.tensor_params,
        )

        self.deep_memory = DeepMemory(
            dataset_or_path=self.path,
            token=self.token,
            logger=self.logger,
            embedding_function=self.embedding_function,
            creds=self.creds,
        )

    def add(
        self,
        embedding_function: Union[Callable, List[Callable]],
        embedding_data: Union[List, List[List]],
        embedding_tensor: Union[str, List[str]],
        return_ids: bool,
        rate_limiter: Dict,
        batch_byte_size: int,
        **tensors,
    ) -> Optional[List[str]]:
        feature_report_path(
            path=self.path,
            feature_name="vs.add",
            parameters={
                "tensors": list(tensors.keys()) if tensors else None,
                "embedding_tensor": embedding_tensor,
                "return_ids": return_ids,
                "embedding_function": True if embedding_function is not None else False,
                "embedding_data": True if embedding_data is not None else False,
                "managed": True,
            },
            token=self.token,
            username=self.username,
        )

        if embedding_function is not None or embedding_data is not None:
            raise NotImplementedError(
                "Embedding function is not supported for ManagedVectorStore. Please send precaculated embeddings."
            )

        (
            embedding_function,
            embedding_data,
            embedding_tensor,
            tensors,
        ) = utils.parse_tensors_kwargs(
            tensors, embedding_function, embedding_data, embedding_tensor
        )

        processed_tensors = {
            t: tensors[t].tolist() if isinstance(tensors[t], np.ndarray) else tensors[t]
            for t in tensors
        }
        utils.check_length_of_each_tensor(processed_tensors)

        response = self.client.vectorstore_add(
            path=self.path,
            processed_tensors=processed_tensors,
            rate_limiter=rate_limiter,
            batch_byte_size=batch_byte_size,
            return_ids=return_ids,
        )

        if return_ids:
            return response.ids

    @use_deep_memory
    def search(
        self,
        embedding_data: Union[str, List[str]],
        embedding_function: Optional[Callable],
        embedding: Union[List[float], np.ndarray],
        k: int,
        distance_metric: str,
        query: str,
        filter: Union[Dict, Callable],
        embedding_tensor: str,
        return_tensors: List[str],
        return_view: bool,
        deep_memory: bool,
    ) -> Union[Dict, Dataset]:
        feature_report_path(
            path=self.path,
            feature_name="vs.search",
            parameters={
                "embedding_data": True if embedding_data is not None else False,
                "embedding_function": True if embedding_function is not None else False,
                "k": k,
                "distance_metric": distance_metric,
                "query": query[0:100] if query is not None else False,
                "filter": True if filter is not None else False,
                "embedding_tensor": embedding_tensor,
                "embedding": True if embedding is not None else False,
                "return_tensors": return_tensors,
                "return_view": return_view,
                "managed": True,
            },
            token=self.token,
            username=self.username,
        )

        if embedding_data is not None or embedding_function is not None:
            raise NotImplementedError(
                "ManagedVectorStore does not support embedding_function search. Please pass a precalculated embedding."
            )

        if filter is not None and not isinstance(filter, dict):
            raise NotImplementedError(
                "Only Filter Dictionary is supported for the ManagedVectorStore."
            )

        if return_view:
            raise NotImplementedError(
                "return_view is not supported for the ManagedVectorStore."
            )

        response = self.client.vectorstore_search(
            path=self.path,
            embedding=embedding,
            k=k,
            distance_metric=distance_metric,
            query=query,
            filter=filter,
            embedding_tensor=embedding_tensor,
            return_tensors=return_tensors,
            deep_memory=deep_memory,
        )
        return response.data

    def delete(
        self,
        row_ids: List[str],
        ids: List[str],
        filter: Union[Dict, Callable],
        query: str,
        delete_all: bool,
    ) -> bool:
        """Delete the data in the Vector Store. Does not delete the tensor definitions. To delete the vector store completely, first run :meth:`VectorStore.delete_by_path()`.

        Examples:
            >>> # Delete using ids:
            >>> data = vector_store.delete(ids)
            >>> # Delete data using filter
            >>> data = vector_store.delete(
            ...        filter = {"json_tensor_name": {"key: value"}, "json_tensor_name_2": {"key_2: value_2"}},
            ... )
            >>> # Delete data using TQL
            >>> data = vector_store.delete(
            ...        query = "select * where ..... <add TQL syntax>",
            ...        exec_option = "compute_engine",
            ... )

        Args:
            ids (Optional[List[str]]): List of unique ids. Defaults to None.
            row_ids (Optional[List[str]]): List of absolute row indices from the dataset. Defaults to None.
            filter (Union[Dict, Callable], optional): Filter for finding samples for deletion.
                - ``Dict`` - Key-value search on tensors of htype json, evaluated on an AND basis (a sample must satisfy all key-value filters to be True) Dict = {"tensor_name_1": {"key": value}, "tensor_name_2": {"key": value}}
                - ``Function`` - Any function that is compatible with `deeplake.filter`.
            query (Optional[str]):  TQL Query string for direct evaluation for finding samples for deletion, without application of additional filters.
            exec_option (Optional[str]): Method for search execution. It could be either ``"python"``, ``"compute_engine"`` or ``"tensor_db"``. Defaults to ``None``, which inherits the option from the Vector Store initialization.
                - ``python`` - Pure-python implementation that runs on the client and can be used for data stored anywhere. WARNING: using this option with big datasets is discouraged because it can lead to memory issues.
                - ``compute_engine`` - Performant C++ implementation of the Deep Lake Compute Engine that runs on the client and can be used for any data stored in or connected to Deep Lake. It cannot be used with in-memory or local datasets.
                - ``tensor_db`` - Performant and fully-hosted Managed Tensor Database that is responsible for storage and query execution. Only available for data stored in the Deep Lake Managed Database. Store datasets in this database by specifying runtime = {"tensor_db": True} during dataset creation.
            delete_all (Optional[bool]): Whether to delete all the samples and version history of the dataset. Defaults to None.

        ..
            # noqa: DAR101

        Returns:
            bool: Returns True if deletion was successful, otherwise it raises a ValueError.

        Raises:
            ValueError: If neither ``ids``, ``filter``, ``query``, nor ``delete_all`` are specified, or if an invalid ``exec_option`` is provided.
        """

        feature_report_path(
            path=self.path,
            feature_name="vs.delete",
            parameters={
                "ids": True if ids is not None else False,
                "row_ids": True if row_ids is not None else False,
                "query": query[0:100] if query is not None else False,
                "filter": True if filter is not None else False,
                "delete_all": delete_all,
                "managed": True,
            },
            token=self.token,
            username=self.username,
        )

        if filter is not None and not isinstance(filter, dict):
            raise NotImplementedError(
                "Only Filter Dictionary is supported for the ManagedVectorStore."
            )

        self.client.vectorstore_remove_rows(
            path=self.path,
            indices=row_ids,
            ids=ids,
            filter=filter,
            query=query,
            delete_all=delete_all,
        )

        return True

    def update_embedding(
        self,
        embedding: Union[List[float], np.ndarray],
        row_ids: List[str],
        ids: List[str],
        filter: Union[Dict, Callable],
        query: str,
    ):
        """Update existing embeddings of the VectorStore, that match either query, filter, ids or row_ids.

        Examples:
            >>> # Update using ids:
            >>> data = vector_store.update(
            ...    ids,
            ...    embedding = [...]
            ... )
            >>> # Update data using filter
            >>> data = vector_store.update(
                    embedding = [...],
            ...     filter = {"json_tensor_name": {"key: value"}, "json_tensor_name_2": {"key_2: value_2"}},
            ... )
            >>> # Update data using TQL, if new embedding function is not specified the embedding_function used
            >>> # during initialization will be used
            >>> data = vector_store.update(
            ...     embedding = [...],
            ...     query = "select * where ..... <add TQL syntax>",
            ... )

        Args:
            row_ids (Optional[List[str]], optional): Row ids of the elements for replacement.
                Defaults to None.
            ids (Optional[List[str]], optional): hash ids of the elements for replacement.
                Defaults to None.
            filter (Optional[Union[Dict, Callable]], optional): Filter for finding samples for replacement.
                - ``Dict`` - Key-value search on tensors of htype json, evaluated on an AND basis (a sample must satisfy all key-value filters to be True) Dict = {"tensor_name_1": {"key": value}, "tensor_name_2": {"key": value}}
                - ``Function`` - Any function that is compatible with `deeplake.filter`
            query (Optional[str], optional): TQL Query string for direct evaluation for finding samples for deletion, without application of additional filters.
                Defaults to None.
        """
        feature_report_path(
            path=self.path,
            feature_name="vs.delete",
            parameters={
                "ids": True if ids is not None else False,
                "row_ids": True if row_ids is not None else False,
                "query": query[0:100] if query is not None else False,
                "filter": True if filter is not None else False,
                "managed": True,
            },
            token=self.token,
            username=self.username,
        )

        if filter is not None and not isinstance(filter, dict):
            raise NotImplementedError(
                "Only Filter Dictionary is supported for the ManagedVectorStore."
            )

        self.client.vectorstore_update_embeddings(
            path=self.path,
            embedding=embedding,
            indices=row_ids,
            ids=ids,
            filter=filter,
            query=query,
        )

    def _get_summary(self):
        """Returns a summary of the Managed Vector Store."""
        return self.client.get_vectorstore_summary(self.path)

    def tensors(self):
        """Returns the list of tensors present in the dataset"""
        return [t["name"] for t in self._get_summary().tensors]

    def summary(self):
        """Prints a summary of the dataset"""
        print(self._get_summary().summary)

    def __len__(self):
        """Length of the dataset"""
        return self._get_summary().length
