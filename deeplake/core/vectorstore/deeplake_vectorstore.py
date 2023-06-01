import logging
from typing import Optional, Any, Iterable, List, Dict, Union, Callable

import numpy as np

try:
    from indra import api  # type: ignore

    _INDRA_INSTALLED = True
except Exception:  # pragma: no cover
    _INDRA_INSTALLED = False  # pragma: no cover

import deeplake
from deeplake.constants import (
    DEFAULT_VECTORSTORE_DEEPLAKE_PATH,
    DEFAULT_VECTORSTORE_TENSORS,
)
from deeplake.core.vectorstore import utils
from deeplake.core.vectorstore.vector_search import vector_search
from deeplake.core.vectorstore.vector_search import dataset as dataset_utils
from deeplake.core.vectorstore.vector_search import filter as filter_utils


logger = logging.getLogger(__name__)


class DeepLakeVectorStore:
    """Base class for DeepLakeVectorStore"""

    def __init__(
        self,
        path: str,
        token: Optional[str] = None,
        tensor_params: List[Dict[str, object]] = DEFAULT_VECTORSTORE_TENSORS,
        embedding_function: Optional[Callable] = None,
        read_only: Optional[bool] = False,
        ingestion_batch_size: int = 1024,
        num_workers: int = 0,
        exec_option: str = "python",
        verbose=True,
        **kwargs: Any,
    ) -> None:
        """DeepLakeVectorStore initialization

        Args:
            path (str): path to the deeplake dataset.
            token (str, optional): Activeloop token, used for fetching credentials for Deep Lake datasets. This is Optional, tokens are normally autogenerated. Defaults to None.
            tensor_params (List[Dict[str, dict]], optional): List of dictionaries that contains information about tensors that user wants to create. Defaults to `DEFAULT_VECTORSTORE_TENSORS`.
            embedding_function (Optional[callable], optional): Function that converts query into embedding. Defaults to None.
            read_only (bool, optional):  Opens dataset in read-only mode if this is passed as True. Defaults to False.
            ingestion_batch_size (int): The batch size to use during ingestion. Defaults to 1024.
            num_workers (int): The number of workers to use for ingesting data in parallel. Defaults to 0.
            exec_option (str): Type of query execution. It could be either "python", "compute_engine" or "tensor_db". Defaults to "python".
                - ``python`` - Pure-python implementation that runs on the client and can be used for data stored anywhere. WARNING: using this option with big datasets is discouraged because it can lead to memory issues.
                - ``compute_engine`` - C++ implementation of the Deep Lake Compute Engine that runs on the client and can be used for any data stored in or connected to Deep Lake. It cannot be used with in-memory or local data.
                - ``tensor_db`` - Fully-hosted Managed Database that is responsible for storage and query execution. Only available for data stored in the Deep Lake Managed Database. This is achieved by specifying runtime = {"tensor_db": True} during dataset creation.
            verbose (bool): Whether to print summary of the dataset created. Defaults to True.
            **kwargs (Any): Additional keyword arguments.
        """
        self.ingestion_batch_size = ingestion_batch_size
        self.num_workers = num_workers
        creds = {"creds": kwargs["creds"]} if "creds" in kwargs else {}
        self.dataset = dataset_utils.create_or_load_dataset(
            tensor_params,
            path,
            token,
            creds,
            logger,
            read_only,
            exec_option,
            embedding_function,
            **kwargs,
        )
        self.embedding_function = embedding_function
        self._exec_option = exec_option
        self.verbose = verbose
        self.tensor_params = tensor_params

    def add(
        self,
        embedding_function: Optional[Callable] = None,
        total_samples_processed: int = 0,
        embedding_data: Optional[List] = None,
        embedding_tensor: Optional[str] = None,
        **tensors,
    ) -> List[str]:
        """Adding elements to deeplake vector store

        Args:
            texts (Iterable[str]): texts to add to deeplake vector store
            embedding_function (callable, optional): embedding function used to convert document texts into embeddings.
            metadatas (List[dict], optional): List of metadatas. Defaults to None.
            ids (List[str], optional): List of document IDs. Defaults to None.
            embeddings (Union[List[float], np.ndarray], optional): embedding of texts. Defaults to None.
            total_samples_processed (int): Total number of samples processed before transforms stopped.

        Returns:
            List[str]: List of document IDs
        """
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

        processed_tensors, id = dataset_utils.preprocess_tensors(
            embedding_data, embedding_tensor, **tensors
        )

        assert id is not None
        utils.check_length_of_each_tensor(processed_tensors)

        dataset_utils.extend_or_ingest_dataset(
            processed_tensors=processed_tensors,
            dataset=self.dataset,
            embedding_function=embedding_function,
            embedding_data=embedding_data,
            embedding_tensor=embedding_tensor,
            ingestion_batch_size=self.ingestion_batch_size,
            num_workers=self.num_workers,
            total_samples_processed=total_samples_processed,
            logger=logger,
        )

        self.dataset.commit(allow_empty=True)
        if self.verbose:
            self.dataset.summary()
        return id

    def search(
        self,
        data_for_embedding=None,
        embedding_function: Optional[Callable] = None,
        embedding: Optional[Union[List[float], np.ndarray]] = None,
        k: int = 4,
        distance_metric: str = "COS",
        query: Optional[str] = None,
        filter: Optional[Union[Dict, Callable]] = None,
        exec_option: Optional[str] = "python",
        embedding_tensor: str = "embedding",
        return_tensors: Optional[List[str]] = None,
        return_view: bool = False,
    ):
        """DeepLakeVectorStore search method that combines embedding search, metadata search, and custom TQL search.

        Examples:
            >>> # Search using an embedding
            >>> data = vector_store.search(
            >>>        embedding = <your_embedding>,
            >>>        exec_option = <preferred_exec_option>,
            >>> )
            >>> # Search using an embedding function and data for embedding
            >>> data = vector_store.search(
            >>>        data_for_embedding = "What does this chatbot do?",
            >>>        embedding_function = <your_embedding_function>,
            >>>        exec_option = <preferred_exec_option>,
            >>> )
            >>> # Add a filter to your search
            >>> data = vector_store.search(
            >>>        embedding = <your_embedding>,
            >>>        exec_option = <preferred_exec_option>,
            >>>        filter = {"json_tensor_name": {"key: value"}, "json_tensor_name_2": {"key_2: value_2"},...}, # Only valid for exec_option = "python"
            >>> )
            >>> # Search using TQL
            >>> data = vector_store.search(
            >>>        query = "select * where ..... <add TQL syntax>",
            >>>        exec_option = <preferred_exec_option>, # Only valid for exec_option = "compute_engine" or "tensor_db"
            >>> )

        Args:
            embedding (Union[np.ndarray, List[float]], optional): Embedding representation for performing the search. Defaults to None. The `data_for_embedding` and `embedding` cannot both be specified.
            data_for_embedding: Data against which the search will be performed by embedding it using the `embedding_function`. Defaults to None. The `data_for_embedding` and `embedding` cannot both be specified.
            embedding_function (callable, optional): function for converting data_for_embedding into embedding. Only valid if data_for_embedding is specified
            k (int): Number of elements to return after running query. Defaults to 4.
            distance_metric (str): Type of distance metric to use for sorting the data. Avaliable options are: "L1", "L2", "COS", "MAX". Defaults to "COS".
            query (Optional[str]):  TQL Query string for direct evaluation, without application of additional filters or vector search.
            filter (Union[Dict, Callable], optional): Additional filter evaluated prior to the embedding search.
                - ``Dict`` - Key-value search on tensors of htype json, evaluated on an AND basis (a sample must satisfy all key-value filters to be True) Dict = {"tensor_name_1": {"key": value}, "tensor_name_2": {"key": value}}
                - ``Function`` - Any function that is compatible with `deeplake.filter`.
            exec_option (Optional[str]): Method for search execution. It could be either "python", "compute_engine" or "tensor_db". Defaults to "python".
                - ``python`` - Pure-python implementation that runs on the client and can be used for data stored anywhere. WARNING: using this option with big datasets is discouraged because it can lead to memory issues.
                - ``compute_engine`` - Performant C++ implementation of the Deep Lake Compute Engine that runs on the client and can be used for any data stored in or connected to Deep Lake. It cannot be used with in-memory or local datasets.
                - ``tensor_db`` - Performant and fully-hosted Managed Tensor Database that is responsible for storage and query execution. Only available for data stored in the Deep Lake Managed Database. Store datasets in this database by specifying runtime = {"db_engine": True} during dataset creation.
            embedding_tensor (str): Name of tensor with embeddings. Defaults to "embedding".
            return_tensors (Optional[List[str]]): List of tensors to return data for. Defaults to None. If None, all tensors are returned.
            return_view (bool): Return a Deep Lake dataset view that satisfied the search parameters, instead of a dictinary with data. Defaults to False.



        Raises:
            ValueError: When invalid parameters are specified.

        Returns:
            Dict: Dictionary where keys are tensor names and values are the results of the search
        """

        exec_option = exec_option or self._exec_option

        self._parse_search_args(
            data_for_embedding=data_for_embedding,
            embedding_function=embedding_function,
            embedding=embedding,
            k=k,
            distance_metric=distance_metric,
            query=query,
            filter=filter,
            exec_option=exec_option,
            embedding_tensor=embedding_tensor,
            return_tensors=return_tensors,
        )

        # if embedding_function is not None or embedding is not None:
        query_emb = dataset_utils.get_embedding(
            embedding,
            data_for_embedding,
            embedding_function=embedding_function,
        )

        if not return_tensors:
            return_tensors = [
                tensor for tensor in self.dataset.tensors if tensor != embedding_tensor
            ]

        return vector_search.search(
            query=query,
            logger=logger,
            filter=filter,
            query_embedding=query_emb,
            k=k,
            distance_metric=distance_metric,
            exec_option=exec_option,
            deeplake_dataset=self.dataset,
            embedding_tensor=embedding_tensor,
            return_tensors=return_tensors,
            return_view=return_view,
        )

    def delete(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[Union[Dict, Callable]] = None,
        query: Optional[str] = None,
        exec_option: Optional[str] = "python",
        delete_all: Optional[bool] = None,
    ) -> bool:
        """Delete the entities in the vector store
        Args:
            ids (Optional[List[str]]): The document_ids to delete.
                Defaults to None.
            filter (Union[Dict, Callable], optional): Filter to select the samples to delete. Defaults to None.
            query (Optional[str]):  TQL Query string for direct evaluation for finding samples for deletion, without application of additional filters.
            exec_option (str, optional): Method for search execution for finding samples for deletion. It could be either "python", "compute_engine". Defaults to "python".
                - ``python`` - Pure-python implementation that runs on the client and can be used for data stored anywhere. WARNING: using this option with big datasets is discouraged because it can lead to memory issues.
                - ``compute_engine`` - Performant C++ implementation of the Deep Lake Compute Engine that runs on the client and can be used for any data stored in or connected to Deep Lake. It cannot be used with in-memory or local datasets.
            delete_all (Optional[bool]): Whether to delete all the samples and version history of the dataset. Defaults to None.
        """

        if ids is None and filter is None and query is None and delete_all is None:
            raise ValueError(
                "Either ids, filter, query, or delete_all must be specified."
            )
        if exec_option not in ("python", "compute_engine", "tensor_db"):
            raise ValueError(
                "Invalid `exec_option` it should be either `python`, `compute_engine`."
            )

        if ids is None:
            (
                self.dataset,
                dataset_deleted,
            ) = dataset_utils.delete_all_samples_if_specified(
                self.dataset,
                delete_all,
            )
            if dataset_deleted:
                return True

            delete_view = self.search(
                filter=filter,
                query=query,
                exec_option=exec_option,
                return_view=True,
                k=int(1e9),
            )

            ids = list(delete_view.sample_indices)

        dataset_utils.delete_and_commit(self.dataset, ids)
        return True

    @staticmethod
    def delete_by_path(path: str) -> None:
        """Force delete dataset by path"""
        deeplake.delete(path, large_ok=True, force=True)

    def tensors(self):
        return self.dataset.tensors

    def summary(self):
        return self.dataset.summary()

    def __len__(self):
        return len(self.dataset)
