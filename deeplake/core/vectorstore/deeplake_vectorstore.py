import logging
import pathlib
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

from deeplake.util.bugout_reporter import feature_report_path, deeplake_reporter


logger = logging.getLogger(__name__)


class DeepLakeVectorStore:
    """Base class for DeepLakeVectorStore"""

    def __init__(
        self,
        path: Union[str, pathlib.Path],
        tensor_params: List[Dict[str, object]] = DEFAULT_VECTORSTORE_TENSORS,
        embedding_function: Optional[Callable] = None,
        read_only: Optional[bool] = False,
        ingestion_batch_size: int = 1024,
        num_workers: int = 0,
        exec_option: str = "python",
        token: Optional[str] = None,
        overwrite: bool = False,
        verbose=True,
        **kwargs: Any,
    ) -> None:
        """Creates an empty DeepLakeVectorStore or loads an existing one if it exists at the specified `path`.

        Examples:
            >>> # Create a vector store with default tensors
            >>> data = DeepLakeVectorStore(
            >>>        path = <path_for_storing_Data>,
            >>> )
            >>> # Create a vector store in the Deep Lake Managed Tensor Database
            >>> data = DeepLakeVectorStore(
            >>>        path = "hub://org_id/dataset_name",
            >>>        runtime = {"tensor_db": True},
            >>> )
            >>> # Create a vector store with custom tensors
            >>> data = DeepLakeVectorStore(
            >>>        path = <path_for_storing_data>,
            >>>        tensor_params = [{"name": "text", "htype": "text"},
            >>>                         {"name": "embedding_1", "htype": "embedding"},
            >>>                         {"name": "embedding_2", "htype": "embedding"},
            >>>                         {"name": "source", "htype": "text"},
            >>>                         {"name": "metadata", "htype": "json"}
            >>>                        ]
            >>> )

        Args:
            path (str, pathlib.Path): - The full path for storing to the Deep Lake Vector Store. It can be:
                - a Deep Lake cloud path of the form ``hub://org_id/dataset_name``. Requires registration with Deep Lake.
                - an s3 path of the form ``s3://bucketname/path/to/dataset``. Credentials are required in either the environment or passed to the creds argument.
                - a local file system path of the form ``./path/to/dataset`` or ``~/path/to/dataset`` or ``path/to/dataset``.
                - a memory path of the form ``mem://path/to/dataset`` which doesn't save the dataset but keeps it in memory instead. Should be used only for testing as it does not persist.
            tensor_params (List[Dict[str, dict]], optional): List of dictionaries that contains information about tensors that user wants to create. See `create_tensor` in Deep Lake API docs for more information. Defaults to `DEFAULT_VECTORSTORE_TENSORS`.
            embedding_function (Optional[callable], optional): Function that converts the embeddable data into embeddings. Defaults to None.
            read_only (bool, optional):  Opens dataset in read-only mode if True. Defaults to False.
            ingestion_batch_size (int): Batch size used during ingestion. Defaults to 1024.
            num_workers (int): The number of workers to use for ingesting data in parallel. Defaults to 0.
            exec_option (str): Default method for search execution. It could be either "python", "compute_engine" or "tensor_db". Defaults to "python".
                - ``python`` - Pure-python implementation that runs on the client and can be used for data stored anywhere. WARNING: using this option with big datasets is discouraged because it can lead to memory issues.
                - ``compute_engine`` - Performant C++ implementation of the Deep Lake Compute Engine that runs on the client and can be used for any data stored in or connected to Deep Lake. It cannot be used with in-memory or local datasets.
                - ``tensor_db`` - Performant and fully-hosted Managed Tensor Database that is responsible for storage and query execution. Only available for data stored in the Deep Lake Managed Database. Store datasets in this database by specifying runtime = {"db_engine": True} during dataset creation.
            token (str, optional): Activeloop token, used for fetching user credentials. This is Optional, tokens are normally autogenerated. Defaults to None.
            overwrite (bool): If set to True this overwrites the Vector Store if it already exists. Defaults to False.
            verbose (bool): Whether to print summary of the dataset created. Defaults to True.
            **kwargs (Any): Additional keyword arguments.

        Danger:
            Setting `overwrite` to `True` will delete all of your data if the Vector Store exists! Be very careful when setting this parameter.
        """
        feature_report_path(
            path,
            "vs.initialize",
            {
                "tensor_params": "default"
                if tensor_params is not None
                else tensor_params,
                "embedding_function": True if embedding_function is not None else False,
                "num_workers": num_workers,
                "overwrite": overwrite,
                "read_only": read_only,
                "ingestion_batch_size": ingestion_batch_size,
                "exec_option": exec_option,
                "token": token,
                "verbose": verbose,
            },
            token=token,
        )

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
            overwrite,
            **kwargs,
        )
        self.embedding_function = embedding_function
        self.exec_option = exec_option
        self.verbose = verbose
        self.tensor_params = tensor_params

    def add(
        self,
        embedding_function: Optional[Callable] = None,
        embedding_data: Optional[List] = None,
        embedding_tensor: Optional[str] = None,
        total_samples_processed: int = 0,
        return_ids: bool = False,
        **tensors,
    ) -> Optional[List[str]]:
        """Adding elements to deeplake vector store

        Examples:
            >>> # initialize vector store:
            >>> deeplake_vector_store = DeepLakeVectorStore(path=<your_dataset_path>)
            >>>
            >>> # if you want to add elements to deeplake_vector_store, you just need to specify the tensors:
            >>> deeplake_vector_store.add(
            >>>     id=id,
            >>>     metadata=metadata,
            >>>     embedding=embedding,
            >>>     text=text,
            >>> )
            >>> # Note if you don't specify id, it will be automatically created
            >>>
            >>> # suppose you want create a dataset with custom tensors and you want to convert data from one tensor into embedding and it to another tensor.
            >>> # to do so, you will need to use the following api:
            >>> deeplake_vector_store = DeepLakeVectorStore(
            >>>     path=<your_dataset_path>,
            >>>     tensors_dict=[
            >>>         {
            >>>             "name": "image_annotations",
            >>>             "htype": "text",
            >>>         },
            >>>         {
            >>>             "name": "image_embeddings",
            >>>             "htype": "embedding",
            >>>         },
            >>>     ]
            >>> )
            >>>
            >>> deeplake_vector_store.add(
            >>>     image_annotations=<some_annotations>,
            >>>     embedding_data=image_annotations # or some other arbitrary data.
            >>>     embedding_tensor="image_embeddings",
            >>> )

        Args:
            embedding_function (Optional[Callable]): embedding function used to convert `embedding_data` into embeddings.
            embedding_data (Optional[List]): Data to be converted into embeddings using the provided `embedding_function`. Defaults to None.
            embedding_tensor (Optional[str]): Tensor where results from the embedding function will be stored. If None, the embedding tensors is automatically inferred (when possible). Defaults to None.
            total_samples_processed (int): Total number of samples processed before ingestion stopped. When specified.
            return_ids (bool): Whether to return added ids as an ouput of the method. Defaults to False.
            **tensors: Keyword arguments where the key is the tensor name, and the value is a list of samples that should be uploaded to that tensor.

        Returns:
            Optional[List[str]]: List of document IDs if `return_ids` is set to True. Otherwise, None.
        """

        deeplake_reporter.feature_report(
            feature_name="vs.add",
            parameters={
                "tensors": list(tensors.keys()) if tensors else None,
                "embedding_tensor": embedding_tensor,
                "total_samples_processed": total_samples_processed,
                "return_ids": return_ids,
                "embedding_function": True if embedding_function is not None else False,
                "embedding_data": True if embedding_data is not None else False,
            },
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

        if return_ids:
            return id
        return None

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
    ) -> Union[Dict, deeplake.core.dataset.Dataset]:
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

        deeplake_reporter.feature_report(
            feature_name="vs.search",
            parameters={
                "data_for_embedding": True if data_for_embedding is not None else False,
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
        )

        exec_option = exec_option or self.exec_option
        embedding_function = embedding_function or self.embedding_function

        utils.parse_search_args(
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
        """Delete the data in the Vector Store. Does not delete the tensor definitions. To delete the vector store completely, first run `DeepLakeVectorStore.delete_by_path()`.

        Examples:
            >>> # Delete using ids:
            >>> data = vector_store.delete(ids)
            >>>
            >>> # Delete data using filter
            >>> data = vector_store.delete(
            >>>        filter = {"json_tensor_name": {"key: value"}, "json_tensor_name_2": {"key_2: value_2"}},
            >>> )
            >>>
            >>> # Delete data using TQL
            >>> data = vector_store.delete(
            >>>        query = "select * where ..... <add TQL syntax>",
            >>>        exec_option = <preferred_exec_option>,
            >>> )

        Args:
            ids (Optional[List[str]]): The document_ids to delete.
                Defaults to None.
            filter (Union[Dict, Callable], optional): Filter for finding samples for deletion.
                - ``Dict`` - Key-value search on tensors of htype json, evaluated on an AND basis (a sample must satisfy all key-value filters to be True) Dict = {"tensor_name_1": {"key": value}, "tensor_name_2": {"key": value}}
                - ``Function`` - Any function that is compatible with `deeplake.filter`.
            query (Optional[str]):  TQL Query string for direct evaluation for finding samples for deletion, without application of additional filters.
            exec_option (str, optional): Method for search execution for finding samples for deletion. It could be either "python", "compute_engine". Defaults to "python".
                - ``python`` - Pure-python implementation that runs on the client and can be used for data stored anywhere. WARNING: using this option with big datasets is discouraged because it can lead to memory issues.
                - ``compute_engine`` - Performant C++ implementation of the Deep Lake Compute Engine that runs on the client and can be used for any data stored in or connected to Deep Lake. It cannot be used with in-memory or local datasets.
            delete_all (Optional[bool]): Whether to delete all the samples and version history of the dataset. Defaults to None.

        Returns:
            bool: Returns True if deletion was successful, otherwise it raises a ValueError.

        Raises:
            ValueError: If neither `ids`, `filter`, `query`, nor `delete_all` are specified, or if an invalid `exec_option` is provided.
        """

        deeplake_reporter.feature_report(
            feature_name="vs.delete",
            parameters={
                "ids": True if ids is not None else False,
                "query": query[0:100] if query is not None else False,
                "filter": True if filter is not None else False,
                "exec_option": exec_option,
                "delete_all": delete_all,
            },
        )

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

            ids = list(delete_view.sample_indices)  # type: ignore

        dataset_utils.delete_and_commit(self.dataset, ids)
        return True

    @staticmethod
    def delete_by_path(
        path: Union[str, pathlib.Path],
        token: Optional[str] = None,
    ) -> None:
        """Deleted the Vector Store at the specified path.

        Args:
            path (str, pathlib.Path): - The full path for storing to the Deep Lake Vector Store.
            token (str, optional): Activeloop token, used for fetching user credentials. This is Optional, tokens are normally autogenerated. Defaults to None.

        Danger:
            This method permanently deletes all of your data in the Vector Store exists! Be very careful when using this method.
        """

        feature_report_path(
            path,
            "vs.delete_by_path",
            {},
            token=token,
        )
        deeplake.delete(path, large_ok=True, token=token, force=True)

    def tensors(self):
        """returns list of tensors in the dataset"""
        return self.dataset.tensors

    def summary(self):
        """prints summary of the dataset"""
        return self.dataset.summary()

    def __len__(self):
        """length of the dataset"""
        return len(self.dataset)
