import logging
import pathlib
from typing import Optional, Any, List, Dict, Union, Callable
import jwt

import numpy as np

import deeplake
from deeplake.core.distance_type import DistanceType
from deeplake.util.dataset import try_flushing
from deeplake.util.path import convert_pathlib_to_string_if_needed

from deeplake.api import dataset
from deeplake.core.dataset import Dataset
from deeplake.constants import (
    DEFAULT_VECTORSTORE_TENSORS,
    MAX_BYTES_PER_MINUTE,
    TARGET_BYTE_SIZE,
    DEFAULT_VECTORSTORE_DISTANCE_METRIC,
    DEFAULT_DEEPMEMORY_DISTANCE_METRIC,
)
from deeplake.client.utils import read_token
from deeplake.core.vectorstore import utils
from deeplake.core.vectorstore.vector_search import vector_search
from deeplake.core.vectorstore.vector_search import dataset as dataset_utils
from deeplake.core.vectorstore.vector_search import filter as filter_utils
from deeplake.core.vectorstore.vector_search.indra import index
from deeplake.util.bugout_reporter import (
    feature_report_path,
)
from deeplake.util.path import get_path_type


logger = logging.getLogger(__name__)


class VectorStore:
    """Base class for VectorStore"""

    def __init__(
        self,
        path: Union[str, pathlib.Path],
        tensor_params: List[Dict[str, object]] = DEFAULT_VECTORSTORE_TENSORS,
        embedding_function: Optional[Callable] = None,
        read_only: Optional[bool] = None,
        ingestion_batch_size: int = 1000,
        index_params: Optional[Dict[str, Union[int, str]]] = None,
        num_workers: int = 0,
        exec_option: str = "auto",
        token: Optional[str] = None,
        overwrite: bool = False,
        verbose: bool = True,
        runtime: Optional[Dict] = None,
        creds: Optional[Union[Dict, str]] = None,
        org_id: Optional[str] = None,
        logger: logging.Logger = logger,
        branch: str = "main",
        **kwargs: Any,
    ) -> None:
        """Creates an empty VectorStore or loads an existing one if it exists at the specified ``path``.

        Examples:
            >>> # Create a vector store with default tensors
            >>> data = VectorStore(
            ...        path = "./my_vector_store",
            ... )
            >>> # Create a vector store in the Deep Lake Managed Tensor Database
            >>> data = VectorStore(
            ...        path = "hub://org_id/dataset_name",
            ...        runtime = {"tensor_db": True},
            ... )
            >>> # Create a vector store with custom tensors
            >>> data = VectorStore(
            ...        path = "./my_vector_store",
            ...        tensor_params = [{"name": "text", "htype": "text"},
            ...                         {"name": "embedding_1", "htype": "embedding"},
            ...                         {"name": "embedding_2", "htype": "embedding"},
            ...                         {"name": "source", "htype": "text"},
            ...                         {"name": "metadata", "htype": "json"}
            ...                        ]
            ... )

        Args:
            path (str, pathlib.Path): - The full path for storing to the Deep Lake Vector Store. It can be:
                - a Deep Lake cloud path of the form ``hub://org_id/dataset_name``. Requires registration with Deep Lake.
                - an s3 path of the form ``s3://bucketname/path/to/dataset``. Credentials are required in either the environment or passed to the creds argument.
                - a local file system path of the form ``./path/to/dataset`` or ``~/path/to/dataset`` or ``path/to/dataset``.
                - a memory path of the form ``mem://path/to/dataset`` which doesn't save the dataset but keeps it in memory instead. Should be used only for testing as it does not persist.
            tensor_params (List[Dict[str, dict]], optional): List of dictionaries that contains information about tensors that user wants to create. See ``create_tensor`` in Deep Lake API docs for more information. Defaults to ``DEFAULT_VECTORSTORE_TENSORS``.
            embedding_function (Optional[callable], optional): Function that converts the embeddable data into embeddings. Input to `embedding_function` is a list of data and output is a list of embeddings. Defaults to None.
            read_only (bool, optional):  Opens dataset in read-only mode if True. Defaults to False.
            num_workers (int): Number of workers to use for parallel ingestion.
            ingestion_batch_size (int): Batch size to use for parallel ingestion.
            index_params (Dict[str, Union[int, str]]): Dictionary containing information about vector index that will be created. Defaults to None, which will utilize ``DEFAULT_VECTORSTORE_INDEX_PARAMS`` from ``deeplake.constants``. The specified key-values override the default ones.
                - threshold: The threshold for the dataset size above which an index will be created for the embedding tensor. When the threshold value is set to -1, index creation is turned off.
                             Defaults to -1, which turns off the index.
                - distance_metric: This key specifies the method of calculating the distance between vectors when creating the vector database (VDB) index. It can either be a string that corresponds to a member of the DistanceType enumeration, or the string value itself.
                    - If no value is provided, it defaults to "L2".
                    - "L2" corresponds to DistanceType.L2_NORM.
                    - "COS" corresponds to DistanceType.COSINE_SIMILARITY.
                - additional_params: Additional parameters for fine-tuning the index.
            exec_option (str): Default method for search execution. It could be either ``"auto"``, ``"python"``, ``"compute_engine"`` or ``"tensor_db"``. Defaults to ``"auto"``. If None, it's set to "auto".
                - ``auto``- Selects the best execution method based on the storage location of the Vector Store. It is the default option.
                - ``python`` - Pure-python implementation that runs on the client and can be used for data stored anywhere. WARNING: using this option with big datasets is discouraged because it can lead to memory issues.
                - ``compute_engine`` - Performant C++ implementation of the Deep Lake Compute Engine that runs on the client and can be used for any data stored in or connected to Deep Lake. It cannot be used with in-memory or local datasets.
                - ``tensor_db`` - Performant and fully-hosted Managed Tensor Database that is responsible for storage and query execution. Only available for data stored in the Deep Lake Managed Database. Store datasets in this database by specifying runtime = {"tensor_db": True} during dataset creation.
            token (str, optional): Activeloop token, used for fetching user credentials. This is Optional, tokens are normally autogenerated. Defaults to None.
            overwrite (bool): If set to True this overwrites the Vector Store if it already exists. Defaults to False.
            verbose (bool): Whether to print summary of the dataset created. Defaults to True.
            creds (dict, str, optional): The string ``ENV`` or a dictionary containing credentials used to access the dataset at the path.
                - If 'aws_access_key_id', 'aws_secret_access_key', 'aws_session_token' are present, these take precedence over credentials present in the environment or in credentials file. Currently only works with s3 paths.
                - It supports 'aws_access_key_id', 'aws_secret_access_key', 'aws_session_token', 'endpoint_url', 'aws_region', 'profile_name' as keys.
                - If 'ENV' is passed, credentials are fetched from the environment variables. This is also the case when creds is not passed for cloud datasets. For datasets connected to hub cloud, specifying 'ENV' will override the credentials fetched from Activeloop and use local ones.
            runtime (Dict, optional): Parameters for creating the Vector Store in Deep Lake's Managed Tensor Database. Not applicable when loading an existing Vector Store. To create a Vector Store in the Managed Tensor Database, set `runtime = {"tensor_db": True}`.
            branch (str): Branch name to use for the Vector Store. Defaults to "main".

            **kwargs (Any): Additional keyword arguments.

        ..
            # noqa: DAR101

        Danger:
            Setting ``overwrite`` to ``True`` will delete all of your data if the Vector Store exists! Be very careful when setting this parameter.
        """
        try:
            from indra import api  # type: ignore

            self.indra_installed = True
        except Exception:  # pragma: no cover
            self.indra_installed = False  # pragma: no cover

        self._token = token
        self.path = convert_pathlib_to_string_if_needed(path)
        self.logger = logger
        self.org_id = org_id if get_path_type(self.path) == "local" else None

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
                "index_params": index_params,
                "exec_option": exec_option,
                "token": self.token,
                "verbose": verbose,
                "runtime": runtime,
            },
            token=self.token,
            username=self.username,
        )

        self.ingestion_batch_size = ingestion_batch_size
        self.index_params = utils.parse_index_params(index_params)
        self.num_workers = num_workers
        self.creds = creds or {}

        self.dataset = dataset_utils.create_or_load_dataset(
            tensor_params,
            path,
            self.token,
            self.creds,
            self.logger,
            read_only,
            exec_option,
            embedding_function,
            overwrite,
            runtime,
            self.org_id,
            branch,
            **kwargs,
        )
        self.embedding_function = embedding_function
        self._exec_option = exec_option
        self.verbose = verbose
        self.tensor_params = tensor_params
        self.distance_metric_index = None
        if utils.index_used(self.exec_option):
            index.index_cache_cleanup(self.dataset)
            self.distance_metric_index = index.validate_and_create_vector_index(
                dataset=self.dataset,
                index_params=self.index_params,
                regenerate_index=False,
            )
        self.deep_memory = None

    @property
    def token(self):
        return self._token or read_token(from_env=True)

    @property
    def exec_option(self) -> str:
        return utils.parse_exec_option(
            self.dataset, self._exec_option, self.indra_installed, self.username
        )

    @property
    def username(self) -> str:
        username = "public"
        if self.token is not None:
            username = jwt.decode(self.token, options={"verify_signature": False})["id"]
        return username

    def add(
        self,
        embedding_function: Optional[Union[Callable, List[Callable]]] = None,
        embedding_data: Optional[Union[List, List[List]]] = None,
        embedding_tensor: Optional[Union[str, List[str]]] = None,
        return_ids: bool = False,
        rate_limiter: Dict = {
            "enabled": False,
            "bytes_per_minute": MAX_BYTES_PER_MINUTE,
        },
        batch_byte_size: int = TARGET_BYTE_SIZE,
        **tensors,
    ) -> Optional[List[str]]:
        """Adding elements to deeplake vector store.

        Tensor names are specified as parameters, and data for each tensor is specified as parameter values. All data must of equal length.

        Examples:
            >>> # Dummy data
            >>> texts = ["Hello", "World"]
            >>> embeddings = [[1, 2, 3], [4, 5, 6]]
            >>> metadatas = [{"timestamp": "01:20"}, {"timestamp": "01:22"}]
            >>> emebdding_fn = lambda x: [[1, 2, 3]] * len(x)
            >>> embedding_fn_2 = lambda x: [[4, 5]] * len(x)
            >>> # Directly upload embeddings
            >>> deeplake_vector_store.add(
            ...     text = texts,
            ...     embedding = embeddings,
            ...     metadata = metadatas,
            ... )
            >>> # Upload embedding via embedding function
            >>> deeplake_vector_store.add(
            ...     text = texts,
            ...     metadata = metadatas,
            ...     embedding_function = embedding_fn,
            ...     embedding_data = texts,
            ... )
            >>> # Upload embedding via embedding function to a user-defined embedding tensor
            >>> deeplake_vector_store.add(
            ...     text = texts,
            ...     metadata = metadatas,
            ...     embedding_function = embedding_fn,
            ...     embedding_data = texts,
            ...     embedding_tensor = "embedding_1",
            ... )
            >>> # Multiple embedding functions (user defined embedding tensors must be specified)
            >>> deeplake_vector_store.add(
            ...     embedding_tensor = ["embedding_1", "embedding_2"]
            ...     embedding_function = [embedding_fn, embedding_fn_2],
            ...     embedding_data = [texts, texts],
            ... )
            >>> # Alternative syntax for multiple embedding functions
            >>> deeplake_vector_store.add(
            ...     text = texts,
            ...     metadata = metadatas,
            ...     embedding_tensor_1 = (embedding_fn, texts),
            ...     embedding_tensor_2 = (embedding_fn_2, texts),
            ... )
            >>> # Add data to fully custom tensors
            >>> deeplake_vector_store.add(
            ...     tensor_A = [1, 2],
            ...     tensor_B = ["a", "b"],
            ...     tensor_C = ["some", "data"],
            ...     embedding_function = embedding_fn,
            ...     embedding_data = texts,
            ...     embedding_tensor = "embedding_1",
            ... )

        Args:
            embedding_function (Optional[Callable]): embedding function used to convert ``embedding_data`` into embeddings. Input to `embedding_function` is a list of data and output is a list of embeddings. Overrides the ``embedding_function`` specified when initializing the Vector Store.
            embedding_data (Optional[List]): Data to be converted into embeddings using the provided ``embedding_function``. Defaults to None.
            embedding_tensor (Optional[str]): Tensor where results from the embedding function will be stored. If None, the embedding tensor is automatically inferred (when possible). Defaults to None.
            return_ids (bool): Whether to return added ids as an ouput of the method. Defaults to False.
            rate_limiter (Dict): Rate limiter configuration. Defaults to ``{"enabled": False, "bytes_per_minute": MAX_BYTES_PER_MINUTE}``.
            batch_byte_size (int): Batch size to use for parallel ingestion. Defaults to ``TARGET_BYTE_SIZE``.
            **tensors: Keyword arguments where the key is the tensor name, and the value is a list of samples that should be uploaded to that tensor.

        Returns:
            Optional[List[str]]: List of ids if ``return_ids`` is set to True. Otherwise, None.
        """

        feature_report_path(
            path=self.path,
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
        data_length = utils.check_length_of_each_tensor(processed_tensors)

        dataset_utils.extend_or_ingest_dataset(
            processed_tensors=processed_tensors,
            dataset=self.dataset,
            embedding_function=embedding_function,
            embedding_data=embedding_data,
            embedding_tensor=embedding_tensor,
            batch_byte_size=batch_byte_size,
            rate_limiter=rate_limiter,
        )

        self._update_index(regenerate_index=data_length > 0)

        try_flushing(self.dataset)

        if self.verbose:
            self.dataset.summary()

        if return_ids:
            return id_
        return None

    def search(
        self,
        embedding_data: Union[str, List[str], None] = None,
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
        deep_memory: bool = False,
    ) -> Union[Dict, Dataset]:
        """VectorStore search method that combines embedding search, metadata search, and custom TQL search.

        Examples:
            >>> # Search using an embedding
            >>> data = vector_store.search(
            ...        embedding = [1, 2, 3],
            ...        exec_option = "python",
            ... )
            >>> # Search using an embedding function and data for embedding
            >>> data = vector_store.search(
            ...        embedding_data = "What does this chatbot do?",
            ...        embedding_function = query_embedding_fn,
            ...        exec_option = "compute_engine",
            ... )
            >>> # Add a filter to your search
            >>> data = vector_store.search(
            ...        embedding = np.ones(3),
            ...        exec_option = "python",
            ...        filter = {"json_tensor_name": {"key: value"}, "json_tensor_name_2": {"key_2: value_2"},...}, # Only valid for exec_option = "python"
            ... )
            >>> # Search using TQL
            >>> data = vector_store.search(
            ...        query = "select * where ..... <add TQL syntax>",
            ...        exec_option = "tensor_db", # Only valid for exec_option = "compute_engine" or "tensor_db"
            ... )

        Args:
            embedding (Union[np.ndarray, List[float]], optional): Embedding representation for performing the search. Defaults to None. The ``embedding_data`` and ``embedding`` cannot both be specified.
            embedding_data (List[str]): Data against which the search will be performed by embedding it using the `embedding_function`. Defaults to None. The `embedding_data` and `embedding` cannot both be specified.
            embedding_function (Optional[Callable], optional): function for converting `embedding_data` into embedding. Only valid if `embedding_data` is specified. Input to `embedding_function` is a list of data and output is a list of embeddings.
            k (int): Number of elements to return after running query. Defaults to 4.
            distance_metric (str): Distance metric to use for sorting the data. Avaliable options are: ``"L1", "L2", "COS", "MAX"``. Defaults to None, which uses same distance metric specified in ``index_params``. If there is no index, it performs linear search using ``DEFAULT_VECTORSTORE_DISTANCE_METRIC``.
            query (Optional[str]):  TQL Query string for direct evaluation, without application of additional filters or vector search.
            filter (Union[Dict, Callable], optional): Additional filter evaluated prior to the embedding search.

                - ``Dict`` - Key-value search on tensors of htype json, evaluated on an AND basis (a sample must satisfy all key-value filters to be True) Dict = {"tensor_name_1": {"key": value}, "tensor_name_2": {"key": value}}
                - ``Function`` - Any function that is compatible with :meth:`Dataset.filter <deeplake.core.dataset.Dataset.filter>`.

            exec_option (Optional[str]): Method for search execution. It could be either ``"python"``, ``"compute_engine"`` or ``"tensor_db"``. Defaults to ``None``, which inherits the option from the Vector Store initialization.

                - ``python`` - Pure-python implementation that runs on the client and can be used for data stored anywhere. WARNING: using this option with big datasets is discouraged because it can lead to memory issues.
                - ``compute_engine`` - Performant C++ implementation of the Deep Lake Compute Engine that runs on the client and can be used for any data stored in or connected to Deep Lake. It cannot be used with in-memory or local datasets.
                - ``tensor_db`` - Performant and fully-hosted Managed Tensor Database that is responsible for storage and query execution. Only available for data stored in the Deep Lake Managed Database. Store datasets in this database by specifying runtime = {"tensor_db": True} during dataset creation.

            embedding_tensor (str): Name of tensor with embeddings. Defaults to "embedding".
            return_tensors (Optional[List[str]]): List of tensors to return data for. Defaults to None, which returns data for all tensors except the embedding tensor (in order to minimize payload). To return data for all tensors, specify return_tensors = "*".
            return_view (bool): Return a Deep Lake dataset view that satisfied the search parameters, instead of a dictionary with data. Defaults to False. If ``True`` return_tensors is set to "*" beucase data is lazy-loaded and there is no cost to including all tensors in the view.
            deep_memory (bool): Whether to use the Deep Memory model for improving search results. Defaults to False if deep_memory is not specified in the Vector Store initialization.
                If True, the distance metric is set to "deepmemory_distance", which represents the metric with which the model was trained. The search is performed using the Deep Memory model. If False, the distance metric is set to "COS" or whatever distance metric user specifies.

        ..
            # noqa: DAR101

        Raises:
            ValueError: When invalid parameters are specified.
            ValueError: when deep_memory is True. Deep Memory is only available for datasets stored in the Deep Lake Managed Database for paid accounts.

        Returns:
            Dict: Dictionary where keys are tensor names and values are the results of the search
        """

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
                "exec_option": exec_option,
                "embedding_tensor": embedding_tensor,
                "embedding": True if embedding is not None else False,
                "return_tensors": return_tensors,
                "return_view": return_view,
            },
            token=self.token,
            username=self.username,
        )

        try_flushing(self.dataset)

        if exec_option is None and self.exec_option != "python" and callable(filter):
            self.logger.warning(
                'Switching exec_option to "python" (runs on client) because filter is specified as a function. '
                f'To continue using the original exec_option "{self.exec_option}", please specify the filter as a dictionary or use the "query" parameter to specify a TQL query.'
            )
            exec_option = "python"

        exec_option = exec_option or self.exec_option

        if deep_memory and not self.deep_memory:
            raise ValueError(
                "Deep Memory is not available for this organization."
                "Deep Memory is only available for waitlisted accounts."
            )

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

        query_emb: Optional[Union[List[float], np.ndarray[Any, Any]]] = None
        if query is None:
            query_emb = dataset_utils.get_embedding(
                embedding,
                embedding_data,
                embedding_function=embedding_function or self.embedding_function,
            )

        if self.distance_metric_index:
            distance_metric = index.parse_index_distance_metric_from_params(
                logger, self.distance_metric_index, distance_metric
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
        row_ids: Optional[List[str]] = None,
        ids: Optional[List[str]] = None,
        filter: Optional[Union[Dict, Callable]] = None,
        query: Optional[str] = None,
        exec_option: Optional[str] = None,
        delete_all: Optional[bool] = None,
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
            return True

        dataset_utils.delete_and_without_commit(self.dataset, row_ids)

        self._update_index(regenerate_index=len(row_ids) > 0 if row_ids else False)

        try_flushing(self.dataset)

        return True

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
        """Recompute existing embeddings of the VectorStore, that match either query, filter, ids or row_ids.

        Examples:
            >>> # Update using ids:
            >>> data = vector_store.update(
            ...    ids,
            ...    embedding_source_tensor = "text",
            ...    embedding_tensor = "embedding",
            ...    embedding_function = embedding_function,
            ... )
            >>> # Update data using filter and several embedding_tensors, several embedding_source_tensors
            >>> # and several embedding_functions:
            >>> data = vector_store.update(
            ...     embedding_source_tensor = ["text", "metadata"],
            ...     embedding_function = ["text_embedding_function", "metadata_embedding_function"],
            ...     filter = {"json_tensor_name": {"key: value"}, "json_tensor_name_2": {"key_2: value_2"}},
            ...     embedding_tensor = ["text_embedding", "metadata_embedding"]
            ... )
            >>> # Update data using TQL, if new embedding function is not specified the embedding_function used
            >>> # during initialization will be used
            >>> data = vector_store.update(
            ...     embedding_source_tensor = "text",
            ...     query = "select * where ..... <add TQL syntax>",
            ...     exec_option = "compute_engine",
            ...     embedding_tensor = "embedding_tensor",
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
            exec_option (Optional[str]): Method for search execution. It could be either ``"python"``, ``"compute_engine"`` or ``"tensor_db"``. Defaults to ``None``, which inherits the option from the Vector Store initialization.
                - ``python`` - Pure-python implementation that runs on the client and can be used for data stored anywhere. WARNING: using this option with big datasets is discouraged because it can lead to memory issues.
                - ``compute_engine`` - Performant C++ implementation of the Deep Lake Compute Engine that runs on the client and can be used for any data stored in or connected to Deep Lake. It cannot be used with in-memory or local datasets.
                - ``tensor_db`` - Performant and fully-hosted Managed Tensor Database that is responsible for storage and query execution. Only available for data stored in the Deep Lake Managed Database. Store datasets in this database by specifying runtime = {"tensor_db": True} during dataset creation.
            embedding_function (Optional[Union[Callable, List[Callable]]], optional): function for converting `embedding_source_tensor` into embedding. Only valid if `embedding_source_tensor` is specified. Defaults to None.
            embedding_source_tensor (Union[str, List[str]], optional): Name of tensor with data that needs to be converted to embeddings. Defaults to `text`.
            embedding_tensor (Optional[Union[str, List[str]]], optional): Name of the tensor with embeddings. Defaults to None.
        """
        feature_report_path(
            path=self.path,
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

        try_flushing(self.dataset)

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

        self._update_index(regenerate_index=len(row_ids) > 0 if row_ids else False)

        try_flushing(self.dataset)

    @staticmethod
    def delete_by_path(
        path: Union[str, pathlib.Path],
        token: Optional[str] = None,
        force: bool = False,
        creds: Optional[Union[Dict, str]] = None,
    ) -> None:
        """Deleted the Vector Store at the specified path.

        Args:
            path (str, pathlib.Path): The full path to the Deep Lake Vector Store.
            token (str, optional): Activeloop token, used for fetching user credentials. This is optional, as tokens are normally autogenerated. Defaults to ``None``.
            creds (dict, str, optional): The string ``ENV`` or a dictionary containing credentials used to access the dataset at the path.
                - If 'aws_access_key_id', 'aws_secret_access_key', 'aws_session_token' are present, these take precedence over credentials present in the environment or in credentials file. Currently only works with s3 paths.
                - It supports 'aws_access_key_id', 'aws_secret_access_key', 'aws_session_token', 'endpoint_url', 'aws_region', 'profile_name' as keys.
                - If 'ENV' is passed, credentials are fetched from the environment variables. This is also the case when creds is not passed for cloud datasets. For datasets connected to hub cloud, specifying 'ENV' will override the credentials fetched from Activeloop and use local ones.
            force (bool): delete the path in a forced manner without rising an exception. Defaults to ``True``.

        Danger:
            This method permanently deletes all of your data if the Vector Store exists! Be very careful when using this method.
        """
        token = token or read_token(from_env=True)

        feature_report_path(
            path,
            "vs.delete_by_path",
            parameters={
                "path": path,
                "token": token,
                "force": force,
                "creds": creds,
            },
            token=token,
        )
        deeplake.delete(path, large_ok=True, token=token, force=force, creds=creds)

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

    def _update_index(self, regenerate_index=False):
        if utils.index_used(self.exec_option):
            index.index_cache_cleanup(self.dataset)
            self.distance_metric_index = index.validate_and_create_vector_index(
                dataset=self.dataset,
                index_params=self.index_params,
                regenerate_index=regenerate_index,
            )


DeepLakeVectorStore = VectorStore
