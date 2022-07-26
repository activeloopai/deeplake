hub.api.dataset
===============
.. currentmodule:: hub.api.dataset
.. class:: dataset
    
    .. staticmethod:: exists(path: Union[str, pathlib.Path], creds: Optional[dict] = None, token: Optional[str] = None) -> bool

        See :func:`hub.exists`.
    
    .. staticmethod:: empty(path: Union[str, pathlib.Path], overwrite: bool = False, public: bool = False, memory_cache_size: int = DEFAULT_MEMORY_CACHE_SIZE, local_cache_size: int = DEFAULT_LOCAL_CACHE_SIZE, creds: Optional[dict] = None, token: Optional[str] = None) -> Dataset

        See :func:`hub.empty`.
    
    .. staticmethod:: load(path: Union[str, pathlib.Path], read_only: Optional[bool] = None, memory_cache_size: int = DEFAULT_MEMORY_CACHE_SIZE, local_cache_size: int = DEFAULT_LOCAL_CACHE_SIZE, creds: Optional[dict] = None, token: Optional[str] = None, verbose: bool = True, access_method: str = "stream") -> Dataset

        See :func:`hub.load`.
    
    .. staticmethod:: rename(old_path: Union[str, pathlib.Path], new_path: Union[str, pathlib.Path], creds: Optional[dict] = None, token: Optional[str] = None) -> Dataset

        See :func:`hub.rename`.
    
    .. staticmethod:: delete(path: Union[str, pathlib.Path], force: bool = False, large_ok: bool = False, creds: Optional[dict] = None, token: Optional[str] = None, verbose: bool = False) -> None

        See :func:`hub.delete`.

    .. staticmethod:: like(dest: Union[str, pathlib.Path], src: Union[str, Dataset, pathlib.Path], tensors: Optional[List[str]] = None, overwrite: bool = False, creds: Optional[dict] = None, token: Optional[str] = None, public: bool = False) -> Dataset
        
        See :func:`hub.like`.
    
    .. staticmethod:: copy(src: Union[str, pathlib.Path, Dataset], dest: Union[str, pathlib.Path], tensors: Optional[List[str]] = None, overwrite: bool = False, src_creds=None, src_token=None, dest_creds=None, dest_token=None, num_workers: int = 0, scheduler="threaded", progressbar=True)
        See :func:`hub.copy`.
    
    .. staticmethod:: deepcopy(src: Union[str, pathlib.Path], dest: Union[str, pathlib.Path], tensors: Optional[List[str]] = None, overwrite: bool = False, src_creds=None, src_token=None, dest_creds=None, dest_token=None, num_workers: int = 0, scheduler="threaded", progressbar=True, public: bool = False, verbose: bool = True)

        See :func:`hub.deepcopy`.
    
    .. staticmethod:: ingest(src: Union[str, pathlib.Path], dest: Union[str, pathlib.Path], images_compression: str = "auto", dest_creds: dict = None, progressbar: bool = True, summary: bool = True, **dataset_kwargs) -> Dataset

        See :func:`hub.ingest`.
    
    .. staticmethod:: ingest_kaggle(tag: str, src: Union[str, pathlib.Path], dest: Union[str, pathlib.Path], exist_ok: bool = False, images_compression: str = "auto", dest_creds: dict = None, kaggle_credentials: dict = None, progressbar: bool = True, summary: bool = True, **dataset_kwargs) -> Dataset

        See :func:`hub.ingest_kaggle`.
    
    .. staticmethod:: ingest_dataframe(src, dest: Union[str, pathlib.Path], dest_creds: Optional[Dict] = None, progressbar: bool = True,**dataset_kwargs)

        See :func:`hub.ingest_dataframe`.
    
    .. staticmethod:: list(workspace: str = "", token: Optional[str] = None,) -> None

        See :func:`hub.list`.