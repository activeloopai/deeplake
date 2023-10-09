deeplake.VectorStore
--------------------

.. autoclass:: deeplake.core.vectorstore.deeplake_vectorstore.VectorStore
   :members:
   :show-inheritance:

   .. automethod:: __init__
      :noindex:

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
