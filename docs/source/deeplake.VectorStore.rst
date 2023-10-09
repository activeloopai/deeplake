deeplake.VectorStore
--------------------

.. autoclass:: deeplake.core.vectorstore.deeplake_vectorstore.VectorStore
   :members:
   :show-inheritance:

   .. automethod:: __init__
      :noindex:
      :template: method

      .. rubric:: Signature

      .. code-block:: python

         __init__(path: Union[str, pathlib.Path],
                  tensor_params: List[Dict[str, object]] = [
                      {'name': 'text', 'htype': 'text', ... },
                      {...},
                      {...},
                      {...}],
                  embedding_function: Optional[Callable] = None,
                  read_only: Optional[bool] = None,
                  ingestion_batch_size: int = 1000,
                  index_params: Optional[Dict[str, Union[int, str]]] = None,
                  num_workers: int = 0,
                  exec_option: str = 'auto',
                  token: Optional[str] = None,
                  overwrite: bool = False,
                  verbose: bool = True,
                  runtime: Optional[Dict] = None,
                  creds: Optional[Union[str, Dict]] = None,
                  org_id: Optional[str] = None,
                  logger: Logger = ...,
                  branch: str = 'main',
                  **kwargs: Any)

      :param path: Path to the vector store.
      :type path: Union[str, pathlib.Path]

      :param tensor_params: Parameters for tensors with default configurations.
      :type tensor_params: List[Dict[str, object]], optional

      :param embedding_function: Function for embeddings. Default is None.
      :type embedding_function: Optional[Callable], optional

      :param read_only: Flag for read-only mode. Default is None.
      :type read_only: Optional[bool], optional

      :param ingestion_batch_size: Batch size for ingestion. Default is 1000.
      :type ingestion_batch_size: int, optional

      :param index_params: Parameters for indexing. Default is None.
      :type index_params: Optional[Dict[str, Union[int, str]]], optional

      :param num_workers: Number of workers. Default is 0.
      :type num_workers: int, optional

      :param exec_option: Execution option. Default is 'auto'.
      :type exec_option: str, optional

      :param token: Token for authentication. Default is None.
      :type token: Optional[str], optional

      :param overwrite: Flag to overwrite existing data. Default is False.
      :type overwrite: bool, optional

      :param verbose: Flag for verbose logging. Default is True.
      :type verbose: bool, optional

      :param runtime: Runtime configurations. Default is None.
      :type runtime: Optional[Dict], optional

      :param creds: Credentials for authentication. Default is None.
      :type creds: Optional[Union[str, Dict]], optional

      :param org_id: Organization ID. Default is None.
      :type org_id: Optional[str], optional

      :param logger: Logger object. Default provided.
      :type logger: Logger, optional

      :param branch: Branch name. Default is 'main'.
      :type branch: str, optional

      :param kwargs: Additional keyword arguments.
      :type kwargs: Any, optional
    
   .. automethod:: add
      :noindex:
      :template: method

      .. rubric:: Signature

      .. code-block:: python

         add(embedding_function: Optional[Union[Callable, List[Callable]]] = None,
             embedding_data: Optional[Union[List, List[List]]] = None,
             embedding_tensor: Optional[Union[str, List[str]]] = None,
             return_ids: bool = False,
             rate_limiter: Dict = {'bytes_per_minute': 1800000.0, 'enabled': False},
             batch_byte_size: int = 10000,
             **tensors) → Optional[List[str]]

      :param embedding_function: Embedding function(s). Default is None.
      :type embedding_function: Optional[Union[Callable, List[Callable]]], optional

      :param embedding_data: Data for embeddings. Default is None.
      :type embedding_data: Optional[Union[List, List[List]]], optional

      :param embedding_tensor: Name of the tensor(s) for embedding. Default is None.
      :type embedding_tensor: Optional[Union[str, List[str]]], optional

      :param return_ids: Flag to return IDs. Default is False.
      :type return_ids: bool, optional

      :param rate_limiter: Rate limiting configuration. Default provided.
      :type rate_limiter: Dict, optional

      :param batch_byte_size: Batch byte size. Default is 10000.
      :type batch_byte_size: int, optional

      :param tensors: Additional tensors.
      :type tensors: Any, optional

   .. automethod:: delete
      :noindex:
      :template: method

      .. rubric:: Signature

      .. code-block:: python

         delete(row_ids: Optional[List[str]] = None,
                ids: Optional[List[str]] = None,
                filter: Optional[Union[Dict, Callable]] = None,
                query: Optional[str] = None,
                exec_option: Optional[str] = None,
                delete_all: Optional[bool] = None) → bool

      :param row_ids: Row IDs to delete. Default is None.
      :type row_ids: Optional[List[str]], optional

      :param ids: IDs to delete. Default is None.
      :type ids: Optional[List[str]], optional

      :param filter: Filter for rows to delete. Can be a dictionary or callable. Default is None.
      :type filter: Optional[Union[Dict, Callable]], optional

      :param query: Query to determine rows to delete. Default is None.
      :type query: Optional[str], optional

      :param exec_option: Execution option for deletion. Default is None.
      :type exec_option: Optional[str], optional

      :param delete_all: Flag to delete all entries. Default is None.
      :type delete_all: Optional[bool], optional

