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
from deeplake.core.vectorstore.vector_search import vector_search
from deeplake.core.vectorstore.vector_search import dataset as dataset_utils
from deeplake.core.vectorstore.vector_search import filter as filter_utils


logger = logging.getLogger(__name__)


class DeepLakeVectorStore:
    """Base class for DeepLakeVectorStore"""

    def __init__(
        self,
        dataset_path: str = DEFAULT_VECTORSTORE_DEEPLAKE_PATH,
        token: Optional[str] = None,
        tensors_dict: List[Dict[str, object]] = DEFAULT_VECTORSTORE_TENSORS,
        embedding_function: Optional[Callable] = None,
        read_only: Optional[bool] = False,
        ingestion_batch_size: int = 1024,
        num_workers: int = 0,
        exec_option: str = "python",
        verbose=False,
        **kwargs: Any,
    ) -> None:
        """DeepLakeVectorStore initialization

        Args:
            dataset_path (str): path to the deeplake dataset. Defaults to DEFAULT_VECTORSTORE_DEEPLAKE_PATH.
            token (str, optional): Activeloop token, used for fetching credentials for Deep Lake datasets. This is Optional, tokens are normally autogenerated. Defaults to None.
            tensors_dict (List[Dict[str, dict]], optional): List of dictionaries that contains information about tensors that user wants to create. Defaults to
            embedding_function (Optional[callable], optional): Function that converts query into embedding. Defaults to None.
            read_only (bool, optional):  Opens dataset in read-only mode if this is passed as True. Defaults to False.
            ingestion_batch_size (int): The batch size to use during ingestion. Defaults to 1024.
            num_workers (int): The number of workers to use for ingesting data in parallel. Defaults to 0.
            exec_option (str): Type of query execution. It could be either "python", "compute_engine" or "tensor_db". Defaults to "python".
                - ``python`` - Pure-python implementation that runs on the client and can be used for data stored anywhere. WARNING: using this option with big datasets is discouraged because it can lead to memory issues.
                - ``compute_engine`` - C++ implementation of the Deep Lake Compute Engine that runs on the client and can be used for any data stored in or connected to Deep Lake. It cannot be used with in-memory or local data.
                - ``tensor_db`` - Fully-hosted Managed Database that is responsible for storage and query execution. Only available for data stored in the Deep Lake Managed Database. This is achieved by specifying runtime = {"tensor_db": True} during dataset creation.
            verbose (bool): Whether to print summary of the dataset created. Defaults to False.
            **kwargs (Any): Additional keyword arguments.
        """
        self.ingestion_batch_size = ingestion_batch_size
        self.num_workers = num_workers
        creds = {"creds": kwargs["creds"]} if "creds" in kwargs else {}
        self.dataset = dataset_utils.create_or_load_dataset(
            tensors_dict,
            dataset_path,
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
        self.tensors_dict = tensors_dict

    def add(
        self,
        embedding_function: Optional[Callable] = None,
        total_samples_processed: int = 0,
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
        processed_tensors, ids = dataset_utils.preprocess_tensors(
            self.tensors_dict, **tensors
        )
        assert ids is not None

        dataset_utils.extend_or_ingest_dataset(
            processed_tensors=processed_tensors,
            dataset=self.dataset,
            embedding_function=embedding_function or self.embedding_function,
            ingestion_batch_size=self.ingestion_batch_size,
            num_workers=self.num_workers,
            total_samples_processed=total_samples_processed,
        )

        self.dataset.commit(allow_empty=True)
        if self.verbose:
            self.dataset.summary()
        return ids

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
            embedding (Union[np.ndarray, List[float]], optional): Embedding representation for performing the search. Defaults to None. The data_for_embedding and embedding cannot both be specified or both be None.
            data_for_embedding (Optional[...], optional): Data against which the search will be performe by embedding it using the embedding_function. Defaults to None. The data_for_embedding and embedding cannot both be specified or both be None.
            embedding_function (callable, optional): function for converting data_for_embedding into embedding. Only valid if data_for_embedding is specified
            k (int): Number of elements to return after running query. Defaults to 4.
            distance_metric (str): Type of distance metric to use for sorting the data. Avaliable options are: "L1", "L2", "COS", "MAX". Defaults to "COS".
            query (Optional[str]):  TQL Query string for direct evaluation, without application of additional filters or vector search. This overrides all other filter-related parameters.
            filter (Union[Dict, Callable], optional): Additional filter evaluated prior to the embedding search.
                - ``Dict`` - Key-value search on tensors of htype json, evaluated on an AND basis (a sample must satisfy all key-value filters to be True) Dict = {"tensor_name_1": {"key": value}, "tensor_name_2": {"key": value}}
                - ``Function`` - Any function that is compatible with `deeplake.filter`.
            exec_option (str, optional): Type of query execution. It could be either "python", "compute_engine" or "tensor_db". Defaults to "python".
                - ``python`` - Pure-python implementation that runs on the client and can be used for data stored anywhere. WARNING: using this option with big datasets is discouraged because it can lead to memory issues.
                - ``compute_engine`` - Performant C++ implementation of the Deep Lake Compute Engine that runs on the client and can be used for any data stored in or connected to Deep Lake. It cannot be used with in-memory or local datasets.
                - ``tensor_db`` - Performant and fully-hosted Managed Tensor Database that is responsible for storage and query execution. Only available for data stored in the Deep Lake Managed Database. Store datasets in this database by specifying runtime = {"db_engine": True} during dataset creation.
            embedding_tensor (str): Name of tensor with embeddings. Defaults to "embedding".
            return_tensors (Optional[List[str]]): List of tensors to return data for. Defaults to None. If None, all tensors are returned.


        Raises:
            ValueError: When invalid execution option is specified

        Returns:
            Dict: Dictionary where keys are tensor names and values are the results of the search
        """

        exec_option = exec_option or self._exec_option
        if exec_option not in ("python", "compute_engine", "tensor_db"):
            raise ValueError(
                "Invalid `exec_option` it should be either `python`, `compute_engine` or `tensor_db`."
            )

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
        )

    def _parse_search_args(self, **kwargs):
        """Helper function for raising errors if invalid parameters are specified to search"""
        if (
            kwargs["data_for_embedding"] is None
            and kwargs["embedding"] is None
            and kwargs["query"] is None
            and kwargs["filter"] is None
        ):
            raise ValueError(
                f"Either a embedding, data_for_embedding, query, or filter must be specified."
            )

        if (
            kwargs["embedding_function"] is None
            and kwargs["embedding"] is None
            and kwargs["query"] is None
        ):
            raise ValueError(
                f"Either an embedding, embedding_function, or query must be specified."
            )

        exec_option = kwargs["exec_option"]
        if exec_option == "python":
            if kwargs["query"] is not None:
                raise ValueError(
                    f"User-specified TQL queries are not support for exec_option={exec_option}."
                )
            if kwargs["query"] is not None:
                raise ValueError(
                    f"query parameter for directly running TQL is invalid for exec_option={exec_option}."
                )
            if kwargs["embedding"] is None and kwargs["embedding_function"] is None:
                raise ValueError(
                    f"Either emebdding or embedding_function must be specified for exec_option={exec_option}."
                )
        else:
            if type(kwargs["filter"]) == Callable:
                raise ValueError(
                    f"UDF filter function are not supported with exec_option={exec_option}"
                )
            if kwargs["query"] and kwargs["filter"]:
                raise ValueError(
                    f"query and filter parameters cannot be specified simultaneously."
                )
            if (
                kwargs["embedding"] is None
                and kwargs["embedding_function"] is None
                and kwargs["query"] is None
            ):
                raise ValueError(
                    f"Either emebdding, embedding_function, or query must be specified for exec_option={exec_option}."
                )
            if kwargs["return_tensors"] and kwargs["query"]:
                raise ValueError(
                    f"return_tensors and query parameters cannot be specified simultaneously, becuase the data that is returned is directly specified in the query."
                )

    def delete(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[Dict[str, str]] = None,
        delete_all: Optional[bool] = None,
    ) -> bool:
        """Delete the entities in the dataset
        Args:
            ids (Optional[List[str]]): The document_ids to delete.
                Defaults to None.
            filter (Optional[Dict[str, str]]): The filter to delete by.
                Defaults to None.
            delete_all (Optional[bool]): Whether to drop the dataset.
                Defaults to None.
        """
        self.dataset, dataset_deleted = dataset_utils.delete_all_samples_if_specified(
            self.dataset, delete_all
        )
        if dataset_deleted:
            return True

        ids = filter_utils.get_converted_ids(self.dataset, filter, ids)
        dataset_utils.delete_and_commit(self.dataset, ids)
        return True

    @staticmethod
    def force_delete_by_path(path: str) -> None:
        """Force delete dataset by path"""
        deeplake.delete(path, large_ok=True, force=True)

    def __len__(self):
        return len(self.dataset)
