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
            creds=creds,
            org_id=org_id,
            logger=logger,
            **kwargs,
        )
        if get_path_type(self.path) != "hub":
            raise ValueError(
                "ManagedVectorStore can only be initialized with a Deep Lake Cloud path."
            )
        self.client = ManagedServiceClient(token=self.token)
        self.client.init_vectorstore(
            path=self.bugout_reporting_path,
            overwrite=overwrite,
            tensor_params=tensor_params,
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
        **tensors,
    ) -> Optional[List[str]]:
        feature_report_path(
            path=self.bugout_reporting_path,
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
        exec_option: Optional[str] = "tensor_db",
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
                "embedding_tensor": embedding_tensor,
                "embedding": True if embedding is not None else False,
                "return_tensors": return_tensors,
                "return_view": return_view,
                "managed": True,
            },
            token=self.token,
            username=self.username,
        )

        if exec_option != "tensor_db":
            raise ValueError("Manged db vectorstore only supports tensor_db execution.")

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

        if exec_option is not None and exec_option != "tensor_db":
            raise ValueError("Manged db vectorstore only supports tensor_db execution.")

        self.client.vectorstore_remove_rows(
            path=self.bugout_reporting_path,
            indices=row_ids,
            ids=ids,
            filter=filter,
            query=query,
            delete_all=delete_all,
        )
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
            path=self.bugout_reporting_path,
            embedding_function=embedding_function,
            embedding_source_tensor=embedding_source_tensor,
            embedding_tensor=embedding_tensor,
            row_ids=row_ids,
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
