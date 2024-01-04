import os

import deeplake
import jwt
import pathlib
import posixpath
import warnings
from typing import Dict, Optional, Union, List

from deeplake.auto.unstructured.kaggle import download_kaggle_dataset
from deeplake.auto.unstructured.image_classification import ImageClassification
from deeplake.auto.unstructured.coco.coco import CocoDataset
from deeplake.auto.unstructured.yolo.yolo import YoloDataset
from deeplake.client.client import DeepLakeBackendClient
from deeplake.client.log import logger
from deeplake.client.utils import get_user_name, read_token
from deeplake.core.dataset import Dataset, dataset_factory
from deeplake.core.tensor import Tensor
from deeplake.core.meta.dataset_meta import DatasetMeta
from deeplake.util.connect_dataset import connect_dataset_entry
from deeplake.util.version_control import (
    load_version_info,
    rebuild_version_info,
    get_parent_and_reset_commit_ids,
    replace_head,
    integrity_check,
)
from deeplake.util.spinner import spinner
from deeplake.util.path import (
    convert_pathlib_to_string_if_needed,
    verify_dataset_name,
    process_dataset_path,
    get_path_type,
)
from deeplake.util.tensor_db import parse_runtime_parameters
from deeplake.hooks import (
    dataset_created,
    dataset_loaded,
    dataset_written,
    dataset_committed,
)
from deeplake.constants import (
    DEFAULT_MEMORY_CACHE_SIZE,
    DEFAULT_LOCAL_CACHE_SIZE,
    DEFAULT_READONLY,
    DATASET_META_FILENAME,
    DATASET_LOCK_FILENAME,
)
from deeplake.util.access_method import (
    check_access_method,
    get_local_dataset,
    parse_access_method,
)
from deeplake.util.auto import get_most_common_extension
from deeplake.util.bugout_reporter import feature_report_path, deeplake_reporter
from deeplake.util.delete_entry import remove_path_from_backend
from deeplake.util.keys import dataset_exists
from deeplake.util.exceptions import (
    AgreementError,
    DatasetHandlerError,
    InvalidFileExtension,
    InvalidPathException,
    PathNotEmptyException,
    SamePathException,
    UserNotLoggedInException,
    TokenPermissionError,
    UnsupportedParameterException,
    DatasetCorruptError,
    CheckoutError,
    ReadOnlyModeError,
    LockedException,
    BadRequestException,
)
from deeplake.util.storage import (
    get_storage_and_cache_chain,
    storage_provider_from_path,
)
from deeplake.util.compute import get_compute_provider
from deeplake.util.remove_cache import get_base_storage
from deeplake.util.cache_chain import generate_chain
from deeplake.core.storage.deeplake_memory_object import DeepLakeMemoryObject


class dataset:
    @staticmethod
    @spinner
    def init(
        path: Union[str, pathlib.Path],
        runtime: Optional[Dict] = None,
        read_only: Optional[bool] = None,
        overwrite: bool = False,
        public: bool = False,
        memory_cache_size: int = DEFAULT_MEMORY_CACHE_SIZE,
        local_cache_size: int = DEFAULT_LOCAL_CACHE_SIZE,
        creds: Optional[Union[Dict, str]] = None,
        token: Optional[str] = None,
        org_id: Optional[str] = None,
        verbose: bool = True,
        access_method: str = "stream",
        unlink: bool = False,
        reset: bool = False,
        check_integrity: Optional[bool] = False,
        lock_enabled: Optional[bool] = True,
        lock_timeout: Optional[int] = 0,
        index_params: Optional[Dict[str, Union[int, str]]] = None,
    ):
        """Returns a :class:`~deeplake.core.dataset.Dataset` object referencing either a new or existing dataset.

        Examples:

            >>> ds = deeplake.dataset("hub://username/dataset")
            >>> ds = deeplake.dataset("s3://mybucket/my_dataset")
            >>> ds = deeplake.dataset("./datasets/my_dataset", overwrite=True)

            Loading to a specfic version:

            >>> ds = deeplake.dataset("hub://username/dataset@new_branch")
            >>> ds = deeplake.dataset("hub://username/dataset@3e49cded62b6b335c74ff07e97f8451a37aca7b2)

            >>> my_commit_id = "3e49cded62b6b335c74ff07e97f8451a37aca7b2"
            >>> ds = deeplake.dataset(f"hub://username/dataset@{my_commit_id}")

        Args:
            path (str, pathlib.Path): - The full path to the dataset. Can be:
                - a Deep Lake cloud path of the form ``hub://username/datasetname``. To write to Deep Lake cloud datasets, ensure that you are logged in to Deep Lake (use 'activeloop login' from command line)
                - an s3 path of the form ``s3://bucketname/path/to/dataset``. Credentials are required in either the environment or passed to the creds argument.
                - a local file system path of the form ``./path/to/dataset`` or ``~/path/to/dataset`` or ``path/to/dataset``.
                - a memory path of the form ``mem://path/to/dataset`` which doesn't save the dataset but keeps it in memory instead. Should be used only for testing as it does not persist.
                - Loading to a specific version:

                    - You can also specify a ``commit_id`` or ``branch`` to load the dataset to that version directly by using the ``@`` symbol.
                    - The path will then be of the form ``hub://username/dataset@{branch}`` or ``hub://username/dataset@{commit_id}``.
                    - See examples above.
            runtime (dict): Parameters for Activeloop DB Engine. Only applicable for hub:// paths.
            read_only (bool, optional): Opens dataset in read only mode if this is passed as ``True``. Defaults to ``False``.
                Datasets stored on Deep Lake cloud that your account does not have write access to will automatically open in read mode.
            overwrite (bool): If set to ``True`` this overwrites the dataset if it already exists. Defaults to ``False``.
            public (bool): Defines if the dataset will have public access. Applicable only if Deep Lake cloud storage is used and a new Dataset is being created. Defaults to ``True``.
            memory_cache_size (int): The size of the memory cache to be used in MB.
            local_cache_size (int): The size of the local filesystem cache to be used in MB.
            creds (dict, str, optional): The string ``ENV`` or a dictionary containing credentials used to access the dataset at the path.
                - If 'aws_access_key_id', 'aws_secret_access_key', 'aws_session_token' are present, these take precedence over credentials present in the environment or in credentials file. Currently only works with s3 paths.
                - It supports 'aws_access_key_id', 'aws_secret_access_key', 'aws_session_token', 'endpoint_url', 'aws_region', 'profile_name' as keys.
                - If 'ENV' is passed, credentials are fetched from the environment variables. This is also the case when creds is not passed for cloud datasets. For datasets connected to hub cloud, specifying 'ENV' will override the credentials fetched from Activeloop and use local ones.
            token (str, optional): Activeloop token, used for fetching credentials to the dataset at path if it is a Deep Lake dataset. This is optional, tokens are normally autogenerated.
            org_id (str, Optional): Organization id to be used for enabling high-performance features. Only applicable for local datasets.
            verbose (bool): If ``True``, logs will be printed. Defaults to ``True``.
            access_method (str): The access method to use for the dataset. Can be:

                    - 'stream'

                        - Streams the data from the dataset i.e. only fetches data when required. This is the default value.

                    - 'download'

                        - Downloads the data to the local filesystem to the path specified in environment variable ``DEEPLAKE_DOWNLOAD_PATH``.
                          This will overwrite ``DEEPLAKE_DOWNLOAD_PATH``.
                        - Raises an exception if ``DEEPLAKE_DOWNLOAD_PATH`` environment variable is not set or if the dataset does not exist.
                        - The 'download' access method can be modified to specify num_workers and/or scheduler.
                          For example: 'download:2:processed' will use 2 workers and use processed scheduler, while 'download:3' will use 3 workers and
                          default scheduler (threaded), and 'download:processed' will use a single worker and use processed scheduler.

                    - 'local'

                        - Downloads the dataset if it doesn't already exist, otherwise loads from local storage.
                        - Raises an exception if ``DEEPLAKE_DOWNLOAD_PATH`` environment variable is not set.
                        - The 'local' access method can be modified to specify num_workers and/or scheduler to be used in case dataset needs to be downloaded.
                          If dataset needs to be downloaded, 'local:2:processed' will use 2 workers and use processed scheduler, while 'local:3' will use 3 workers
                          and default scheduler (threaded), and 'local:processed' will use a single worker and use processed scheduler.
            unlink (bool): Downloads linked samples if set to ``True``. Only applicable if ``access_method`` is ``download`` or ``local``. Defaults to ``False``.
            reset (bool): If the specified dataset cannot be loaded due to a corrupted HEAD state of the branch being loaded,
                          setting ``reset=True`` will reset HEAD changes and load the previous version.
            check_integrity (bool, Optional): Performs an integrity check by default (None) if the dataset has 20 or fewer tensors.
                                              Set to ``True`` to force integrity check, ``False`` to skip integrity check.
            lock_timeout (int): Number of seconds to wait before throwing a LockException. If None, wait indefinitely
            lock_enabled (bool): If true, the dataset manages a write lock. NOTE: Only set to False if you are managing concurrent access externally
            index_params: Optional[Dict[str, Union[int, str]]] = None : The index parameters used while creating vector store is passed down to dataset.

        ..
            # noqa: DAR101

        Returns:
            Dataset: Dataset created using the arguments provided.

        Raises:
            AgreementError: When agreement is rejected
            UserNotLoggedInException: When user is not logged in
            InvalidTokenException: If the specified token is invalid
            TokenPermissionError: When there are permission or other errors related to token
            CheckoutError: If version address specified in the path cannot be found
            DatasetCorruptError: If loading the dataset failed due to corruption and ``reset`` is not ``True``
            ValueError: If version is specified in the path when creating a dataset or If the org id is provided but dataset is ot local, or If the org id is provided but dataset is ot local
            ReadOnlyModeError: If reset is attempted in read-only mode
            LockedException: When attempting to open a dataset for writing when it is locked by another machine
            DatasetHandlerError: If overwriting the dataset fails
            Exception: Re-raises caught exception if reset cannot fix the issue

        Danger:
            Setting ``overwrite`` to ``True`` will delete all of your data if it exists! Be very careful when setting this parameter.

        Warning:
            Setting ``access_method`` to download will overwrite the local copy of the dataset if it was previously downloaded.

        Note:
            Any changes made to the dataset in download / local mode will only be made to the local copy and will not be reflected in the original dataset.
        """
        access_method, num_workers, scheduler = parse_access_method(access_method)
        check_access_method(access_method, overwrite, unlink)

        path, address = process_dataset_path(path)
        verify_dataset_name(path)

        if org_id is not None and get_path_type(path) != "local":
            raise ValueError("org_id parameter can only be used with local datasets")

        if creds is None:
            creds = {}

        db_engine = parse_runtime_parameters(path, runtime)["tensor_db"]

        try:
            storage, cache_chain = get_storage_and_cache_chain(
                path=path,
                db_engine=db_engine,
                read_only=read_only,
                creds=creds,
                token=token,
                memory_cache_size=memory_cache_size,
                local_cache_size=local_cache_size,
            )

            feature_report_path(path, "dataset", {"Overwrite": overwrite}, token=token)
        except Exception as e:
            if isinstance(e, UserNotLoggedInException):
                raise UserNotLoggedInException from None
            raise
        ds_exists = dataset_exists(cache_chain)

        if ds_exists:
            if overwrite:
                try:
                    cache_chain.clear()
                except Exception as e:
                    raise DatasetHandlerError(
                        "Dataset overwrite failed. See traceback for more information."
                    ) from e
                create = True
            else:
                create = False
        else:
            create = True

        if create and address:
            raise ValueError(
                "deeplake.dataset does not accept version address when writing a dataset."
            )

        dataset_kwargs: Dict[str, Union[None, str, bool, int, Dict]] = {
            "path": path,
            "read_only": read_only,
            "token": token,
            "org_id": org_id,
            "verbose": verbose,
            "lock_timeout": lock_timeout,
            "lock_enabled": lock_enabled,
            "index_params": index_params,
        }

        if access_method == "stream":
            dataset_kwargs.update(
                {
                    "address": address,
                    "storage": cache_chain,
                    "public": public,
                }
            )
        else:
            dataset_kwargs.update(
                {
                    "access_method": access_method,
                    "memory_cache_size": memory_cache_size,
                    "local_cache_size": local_cache_size,
                    "creds": creds,
                    "ds_exists": ds_exists,
                    "num_workers": num_workers,
                    "scheduler": scheduler,
                    "reset": reset,
                    "unlink": unlink,
                }
            )

        try:
            return dataset._load(
                dataset_kwargs, access_method, create, check_integrity=check_integrity
            )
        except (AgreementError, CheckoutError, LockedException) as e:
            raise e from None
        except Exception as e:
            if create:
                raise e
            if access_method == "stream":
                if not reset:
                    if isinstance(e, DatasetCorruptError):
                        raise DatasetCorruptError(
                            message=e.message,
                            action="Try using `reset=True` to reset HEAD changes and load the previous commit.",
                            cause=e.__cause__,
                        )
                    raise DatasetCorruptError(
                        "Exception occurred (see Traceback). The dataset maybe corrupted. "
                        "Try using `reset=True` to reset HEAD changes and load the previous commit."
                    ) from e
                return dataset._reset_and_load(
                    cache_chain, access_method, dataset_kwargs, address, e
                )
            raise e

    @staticmethod
    def exists(
        path: Union[str, pathlib.Path],
        creds: Optional[Union[Dict, str]] = None,
        token: Optional[str] = None,
    ) -> bool:
        """Checks if a dataset exists at the given ``path``.

        Args:
            path (str, pathlib.Path): the path which needs to be checked.
            creds (dict, str, optional): The string ``ENV`` or a dictionary containing credentials used to access the dataset at the path.
                - If 'aws_access_key_id', 'aws_secret_access_key', 'aws_session_token' are present, these take precedence over credentials present in the environment or in credentials file. Currently only works with s3 paths.
                - It supports 'aws_access_key_id', 'aws_secret_access_key', 'aws_session_token', 'endpoint_url', 'aws_region', 'profile_name' as keys.
                - If 'ENV' is passed, credentials are fetched from the environment variables. This is also the case when creds is not passed for cloud datasets. For datasets connected to hub cloud, specifying 'ENV' will override the credentials fetched from Activeloop and use local ones.
            token (str, optional): Activeloop token, used for fetching credentials to the dataset at path if it is a Deep Lake dataset. This is optional, tokens are normally autogenerated.

        Returns:
            A boolean confirming whether the dataset exists or not at the given path.

        Raises:
            ValueError: If version is specified in the path
        """
        path, address = process_dataset_path(path)

        if address:
            raise ValueError(
                "deeplake.exists does not accept version address in the dataset path."
            )

        if creds is None:
            creds = {}
        try:
            storage, cache_chain = get_storage_and_cache_chain(
                path=path,
                read_only=True,
                creds=creds,
                token=token,
                memory_cache_size=DEFAULT_MEMORY_CACHE_SIZE,
                local_cache_size=DEFAULT_LOCAL_CACHE_SIZE,
            )
        except TokenPermissionError:
            # Cloud Dataset does not exist
            return False
        return dataset_exists(storage)

    @staticmethod
    def empty(
        path: Union[str, pathlib.Path],
        runtime: Optional[dict] = None,
        overwrite: bool = False,
        public: bool = False,
        memory_cache_size: int = DEFAULT_MEMORY_CACHE_SIZE,
        local_cache_size: int = DEFAULT_LOCAL_CACHE_SIZE,
        creds: Optional[Union[Dict, str]] = None,
        token: Optional[str] = None,
        org_id: Optional[str] = None,
        lock_enabled: Optional[bool] = True,
        lock_timeout: Optional[int] = 0,
        verbose: bool = True,
        index_params: Optional[Dict[str, Union[int, str]]] = None,
    ) -> Dataset:
        """Creates an empty dataset

        Args:
            path (str, pathlib.Path): - The full path to the dataset. It can be:
                - a Deep Lake cloud path of the form ``hub://org_id/dataset_name``. Requires registration with Deep Lake.
                - an s3 path of the form ``s3://bucketname/path/to/dataset``. Credentials are required in either the environment or passed to the creds argument.
                - a local file system path of the form ``./path/to/dataset`` or ``~/path/to/dataset`` or ``path/to/dataset``.
                - a memory path of the form ``mem://path/to/dataset`` which doesn't save the dataset but keeps it in memory instead. Should be used only for testing as it does not persist.
            runtime (dict): Parameters for creating a dataset in the Deep Lake Tensor Database. Only applicable for paths of the form ``hub://org_id/dataset_name`` and runtime  must be ``{"tensor_db": True}``.
            overwrite (bool): If set to ``True`` this overwrites the dataset if it already exists. Defaults to ``False``.
            public (bool): Defines if the dataset will have public access. Applicable only if Deep Lake cloud storage is used and a new Dataset is being created. Defaults to ``False``.
            memory_cache_size (int): The size of the memory cache to be used in MB.
            local_cache_size (int): The size of the local filesystem cache to be used in MB.
            creds (dict, str, optional): The string ``ENV`` or a dictionary containing credentials used to access the dataset at the path.
                - If 'aws_access_key_id', 'aws_secret_access_key', 'aws_session_token' are present, these take precedence over credentials present in the environment or in credentials file. Currently only works with s3 paths.
                - It supports 'aws_access_key_id', 'aws_secret_access_key', 'aws_session_token', 'endpoint_url', 'aws_region', 'profile_name' as keys.
                - If 'ENV' is passed, credentials are fetched from the environment variables. This is also the case when creds is not passed for cloud datasets. For datasets connected to hub cloud, specifying 'ENV' will override the credentials fetched from Activeloop and use local ones.
            token (str, optional): Activeloop token, used for fetching credentials to the dataset at path if it is a Deep Lake dataset. This is optional, tokens are normally autogenerated.
            org_id (str, Optional): Organization id to be used for enabling high-performance features. Only applicable for local datasets.
            verbose (bool): If True, logs will be printed. Defaults to True.
            lock_timeout (int): Number of seconds to wait before throwing a LockException. If None, wait indefinitely
            lock_enabled (bool): If true, the dataset manages a write lock. NOTE: Only set to False if you are managing concurrent access externally.
            index_params: Optional[Dict[str, Union[int, str]]]: Index parameters used while creating vector store, passed down to dataset.

        Returns:
            Dataset: Dataset created using the arguments provided.

        Raises:
            DatasetHandlerError: If a Dataset already exists at the given path and overwrite is False.
            UserNotLoggedInException: When user is not logged in
            InvalidTokenException: If the specified toke is invalid
            TokenPermissionError: When there are permission or other errors related to token
            ValueError: If version is specified in the path

        Danger:
            Setting ``overwrite`` to ``True`` will delete all of your data if it exists! Be very careful when setting this parameter.
        """
        path, address = process_dataset_path(path)

        if org_id is not None and get_path_type(path) != "local":
            raise ValueError("org_id parameter can only be used with local datasets")
        db_engine = parse_runtime_parameters(path, runtime)["tensor_db"]

        if address:
            raise ValueError(
                "deeplake.empty does not accept version address in the dataset path."
            )

        verify_dataset_name(path)

        if creds is None:
            creds = {}

        try:
            storage, cache_chain = get_storage_and_cache_chain(
                path=path,
                db_engine=db_engine,
                read_only=False,
                creds=creds,
                token=token,
                memory_cache_size=memory_cache_size,
                local_cache_size=local_cache_size,
            )

            feature_report_path(
                path,
                "empty",
                {
                    "runtime": runtime,
                    "overwrite": overwrite,
                    "lock_enabled": lock_enabled,
                    "lock_timeout": lock_timeout,
                    "index_params": index_params,
                },
                token=token,
            )
        except Exception as e:
            if isinstance(e, UserNotLoggedInException):
                raise UserNotLoggedInException from None
            raise

        if overwrite and dataset_exists(cache_chain):
            try:
                cache_chain.clear()
            except Exception as e:
                raise DatasetHandlerError(
                    "Dataset overwrite failed. See traceback for more information."
                ) from e
        elif dataset_exists(cache_chain):
            raise DatasetHandlerError(
                f"A dataset already exists at the given path ({path}). If you want to create"
                f" a new empty dataset, either specify another path or use overwrite=True. "
                f"If you want to load the dataset that exists at this path, use deeplake.load() instead."
            )

        dataset_kwargs = {
            "path": path,
            "storage": cache_chain,
            "read_only": storage.read_only,
            "public": public,
            "token": token,
            "org_id": org_id,
            "verbose": verbose,
            "lock_timeout": lock_timeout,
            "lock_enabled": lock_enabled,
            "index_params": index_params,
        }
        ret = dataset._load(dataset_kwargs, create=True)
        return ret

    @staticmethod
    @spinner
    def load(
        path: Union[str, pathlib.Path],
        read_only: Optional[bool] = None,
        memory_cache_size: int = DEFAULT_MEMORY_CACHE_SIZE,
        local_cache_size: int = DEFAULT_LOCAL_CACHE_SIZE,
        creds: Optional[Union[dict, str]] = None,
        token: Optional[str] = None,
        org_id: Optional[str] = None,
        verbose: bool = True,
        access_method: str = "stream",
        unlink: bool = False,
        reset: bool = False,
        check_integrity: Optional[bool] = None,
        lock_timeout: Optional[int] = 0,
        lock_enabled: Optional[bool] = True,
        index_params: Optional[Dict[str, Union[int, str]]] = None,
    ) -> Dataset:
        """Loads an existing dataset

        Examples:

            >>> ds = deeplake.load("hub://username/dataset")
            >>> ds = deeplake.load("s3://mybucket/my_dataset")
            >>> ds = deeplake.load("./datasets/my_dataset", overwrite=True)

            Loading to a specfic version:

            >>> ds = deeplake.load("hub://username/dataset@new_branch")
            >>> ds = deeplake.load("hub://username/dataset@3e49cded62b6b335c74ff07e97f8451a37aca7b2)

            >>> my_commit_id = "3e49cded62b6b335c74ff07e97f8451a37aca7b2"
            >>> ds = deeplake.load(f"hub://username/dataset@{my_commit_id}")

        Args:
            path (str, pathlib.Path): - The full path to the dataset. Can be:
                - a Deep Lake cloud path of the form ``hub://username/datasetname``. To write to Deep Lake cloud datasets, ensure that you are logged in to Deep Lake (use 'activeloop login' from command line)
                - an s3 path of the form ``s3://bucketname/path/to/dataset``. Credentials are required in either the environment or passed to the creds argument.
                - a local file system path of the form ``./path/to/dataset`` or ``~/path/to/dataset`` or ``path/to/dataset``.
                - a memory path of the form ``mem://path/to/dataset`` which doesn't save the dataset but keeps it in memory instead. Should be used only for testing as it does not persist.
                - Loading to a specific version:

                        - You can also specify a ``commit_id`` or ``branch`` to load the dataset to that version directly by using the ``@`` symbol.
                        - The path will then be of the form ``hub://username/dataset@{branch}`` or ``hub://username/dataset@{commit_id}``.
                        - See examples above.
            read_only (bool, optional): Opens dataset in read only mode if this is passed as ``True``. Defaults to ``False``.
                Datasets stored on Deep Lake cloud that your account does not have write access to will automatically open in read mode.
            memory_cache_size (int): The size of the memory cache to be used in MB.
            local_cache_size (int): The size of the local filesystem cache to be used in MB.
            creds (dict, str, optional): The string ``ENV`` or a dictionary containing credentials used to access the dataset at the path.
                - If 'aws_access_key_id', 'aws_secret_access_key', 'aws_session_token' are present, these take precedence over credentials present in the environment or in credentials file. Currently only works with s3 paths.
                - It supports 'aws_access_key_id', 'aws_secret_access_key', 'aws_session_token', 'endpoint_url', 'aws_region', 'profile_name' as keys.
                - If 'ENV' is passed, credentials are fetched from the environment variables. This is also the case when creds is not passed for cloud datasets. For datasets connected to hub cloud, specifying 'ENV' will override the credentials fetched from Activeloop and use local ones.
            token (str, optional): Activeloop token, used for fetching credentials to the dataset at path if it is a Deep Lake dataset. This is optional, tokens are normally autogenerated.
            org_id (str, Optional): Organization id to be used for enabling high-performance features. Only applicable for local datasets.
            verbose (bool): If ``True``, logs will be printed. Defaults to ``True``.
            access_method (str): The access method to use for the dataset. Can be:

                    - 'stream'

                        - Streams the data from the dataset i.e. only fetches data when required. This is the default value.

                    - 'download'

                        - Downloads the data to the local filesystem to the path specified in environment variable ``DEEPLAKE_DOWNLOAD_PATH``.
                          This will overwrite ``DEEPLAKE_DOWNLOAD_PATH``.
                        - Raises an exception if ``DEEPLAKE_DOWNLOAD_PATH`` environment variable is not set or if the dataset does not exist.
                        - The 'download' access method can be modified to specify num_workers and/or scheduler.
                          For example: 'download:2:processed' will use 2 workers and use processed scheduler, while 'download:3' will use 3 workers and
                          default scheduler (threaded), and 'download:processed' will use a single worker and use processed scheduler.

                    - 'local'

                        - Downloads the dataset if it doesn't already exist, otherwise loads from local storage.
                        - Raises an exception if ``DEEPLAKE_DOWNLOAD_PATH`` environment variable is not set.
                        - The 'local' access method can be modified to specify num_workers and/or scheduler to be used in case dataset needs to be downloaded.
                          If dataset needs to be downloaded, 'local:2:processed' will use 2 workers and use processed scheduler, while 'local:3' will use 3 workers
                          and default scheduler (threaded), and 'local:processed' will use a single worker and use processed scheduler.
            unlink (bool): Downloads linked samples if set to ``True``. Only applicable if ``access_method`` is ``download`` or ``local``. Defaults to ``False``.
            reset (bool): If the specified dataset cannot be loaded due to a corrupted HEAD state of the branch being loaded,
                          setting ``reset=True`` will reset HEAD changes and load the previous version.
            check_integrity (bool, Optional): Performs an integrity check by default (None) if the dataset has 20 or fewer tensors.
                                              Set to ``True`` to force integrity check, ``False`` to skip integrity check.

        ..
            # noqa: DAR101

        Returns:
            Dataset: Dataset loaded using the arguments provided.

        Raises:
            DatasetHandlerError: If a Dataset does not exist at the given path.
            AgreementError: When agreement is rejected
            UserNotLoggedInException: When user is not logged in
            InvalidTokenException: If the specified toke is invalid
            TokenPermissionError: When there are permission or other errors related to token
            CheckoutError: If version address specified in the path cannot be found
            DatasetCorruptError: If loading the dataset failed due to corruption and ``reset`` is not ``True``
            ReadOnlyModeError: If reset is attempted in read-only mode
            LockedException: When attempting to open a dataset for writing when it is locked by another machine
            ValueError: If ``org_id`` is specified for a non-local dataset
            Exception: Re-raises caught exception if reset cannot fix the issue
            ValueError: If the org id is provided but the dataset is not local

        Warning:
            Setting ``access_method`` to download will overwrite the local copy of the dataset if it was previously downloaded.

        Note:
            Any changes made to the dataset in download / local mode will only be made to the local copy and will not be reflected in the original dataset.
        """
        access_method, num_workers, scheduler = parse_access_method(access_method)
        check_access_method(access_method, overwrite=False, unlink=unlink)

        path, address = process_dataset_path(path)

        if creds is None:
            creds = {}

        if org_id is not None and get_path_type(path) != "local":
            raise ValueError("org_id parameter can only be used with local datasets")

        try:
            storage, cache_chain = get_storage_and_cache_chain(
                path=path,
                read_only=read_only,
                creds=creds,
                token=token,
                memory_cache_size=memory_cache_size,
                local_cache_size=local_cache_size,
            )
            feature_report_path(
                path,
                "load",
                {
                    "lock_enabled": lock_enabled,
                    "lock_timeout": lock_timeout,
                    "index_params": index_params,
                },
                token=token,
            )
        except Exception as e:
            if isinstance(e, UserNotLoggedInException):
                raise UserNotLoggedInException from None
            raise
        if not dataset_exists(cache_chain):
            raise DatasetHandlerError(
                f"A Deep Lake dataset does not exist at the given path ({path}). Check the path provided or in case you want to create a new dataset, use deeplake.empty()."
            )

        dataset_kwargs: Dict[str, Union[None, str, bool, int, Dict]] = {
            "path": path,
            "read_only": read_only,
            "token": token,
            "org_id": org_id,
            "verbose": verbose,
            "lock_timeout": lock_timeout,
            "lock_enabled": lock_enabled,
            "index_params": index_params,
        }

        if access_method == "stream":
            dataset_kwargs.update(
                {
                    "address": address,
                    "storage": cache_chain,
                }
            )
        else:
            dataset_kwargs.update(
                {
                    "access_method": access_method,
                    "memory_cache_size": memory_cache_size,
                    "local_cache_size": local_cache_size,
                    "creds": creds,
                    "ds_exists": True,
                    "num_workers": num_workers,
                    "scheduler": scheduler,
                    "reset": reset,
                    "unlink": unlink,
                }
            )

        try:
            return dataset._load(
                dataset_kwargs, access_method, check_integrity=check_integrity
            )
        except (AgreementError, CheckoutError, LockedException) as e:
            raise e from None
        except Exception as e:
            if access_method == "stream":
                if not reset:
                    if isinstance(e, DatasetCorruptError):
                        raise DatasetCorruptError(
                            message=e.message,
                            action="Try using `reset=True` to reset HEAD changes and load the previous commit.",
                            cause=e.__cause__,
                        )
                    raise DatasetCorruptError(
                        "Exception occurred (see Traceback). The dataset maybe corrupted. "
                        "Try using `reset=True` to reset HEAD changes and load the previous commit. "
                        "This will delete all uncommitted changes on the branch you are trying to load."
                    ) from e
                return dataset._reset_and_load(
                    cache_chain, access_method, dataset_kwargs, address, e
                )
            raise e

    @staticmethod
    def _reset_and_load(storage, access_method, dataset_kwargs, address, err):
        """Reset and then load the dataset. Only called when loading dataset errored out with ``err``."""
        if access_method != "stream":
            dataset_kwargs["reset"] = True
            ds = dataset._load(dataset_kwargs, access_method)
            return ds

        try:
            version_info = load_version_info(storage)
        except Exception:
            raise err

        address = address or "main"
        parent_commit_id, reset_commit_id = get_parent_and_reset_commit_ids(
            version_info, address
        )
        if parent_commit_id is False:
            # non-head node corrupted
            raise err
        if storage.read_only:
            msg = "Cannot reset when loading dataset in read-only mode."
            if parent_commit_id:
                msg += " However, you can try loading the previous commit using "
                msg += f"`deeplake.load('{dataset_kwargs.get('path')}@{parent_commit_id}')`."
            raise ReadOnlyModeError(msg)
        if parent_commit_id is None:
            # no commits in the dataset
            storage.clear()
            ds = dataset._load(dataset_kwargs, access_method)
            return ds

        # load previous version, replace head and checkout to new head
        dataset_kwargs["address"] = parent_commit_id
        ds = dataset._load(dataset_kwargs, access_method)
        new_commit_id = replace_head(storage, ds.version_state, reset_commit_id)
        ds.checkout(new_commit_id)

        current_node = ds.version_state["commit_node_map"][ds.commit_id]
        verbose = dataset_kwargs.get("verbose")
        if verbose:
            logger.info(f"HEAD reset. Current version:\n{current_node}")
        return ds

    @staticmethod
    def _load(dataset_kwargs, access_method=None, create=False, check_integrity=None):
        if access_method in ("stream", None):
            ret = dataset_factory(**dataset_kwargs)
            if create:
                dataset_created(ret)
            else:
                dataset_loaded(ret)

            if check_integrity is None:
                if len(ret.meta.tensors) < 20:
                    check_integrity = True
                else:
                    warnings.warn(
                        "Dataset has more than 20 tensors. Skipping integrity check. Specify `check_integrity=True` to perform integrity check."
                    )
                    check_integrity = False

            if check_integrity:
                integrity_check(ret)

            verbose = dataset_kwargs.get("verbose")
            path = dataset_kwargs.get("path")
            if verbose:
                logger.info(f"{path} loaded successfully.")
        else:
            ret = get_local_dataset(**dataset_kwargs)
        return ret

    @staticmethod
    def rename(
        old_path: Union[str, pathlib.Path],
        new_path: Union[str, pathlib.Path],
        creds: Optional[Union[dict, str]] = None,
        token: Optional[str] = None,
    ) -> Dataset:
        """Renames dataset at ``old_path`` to ``new_path``.

        Examples:

            >>> deeplake.rename("hub://username/image_ds", "hub://username/new_ds")
            >>> deeplake.rename("s3://mybucket/my_ds", "s3://mybucket/renamed_ds")

        Args:
            old_path (str, pathlib.Path): The path to the dataset to be renamed.
            new_path (str, pathlib.Path): Path to the dataset after renaming.
            creds (dict, str, optional): The string ``ENV`` or a dictionary containing credentials used to access the dataset at the path.
                - If 'aws_access_key_id', 'aws_secret_access_key', 'aws_session_token' are present, these take precedence over credentials present in the environment or in credentials file. Currently only works with s3 paths.
                - It supports 'aws_access_key_id', 'aws_secret_access_key', 'aws_session_token', 'endpoint_url', 'aws_region', 'profile_name' as keys.
                - If 'ENV' is passed, credentials are fetched from the environment variables. This is also the case when creds is not passed for cloud datasets. For datasets connected to hub cloud, specifying 'ENV' will override the credentials fetched from Activeloop and use local ones.
            token (str, optional): Activeloop token, used for fetching credentials to the dataset at path if it is a Deep Lake dataset. This is optional, tokens are normally autogenerated.

        Returns:
            Dataset: The renamed Dataset.

        Raises:
            DatasetHandlerError: If a Dataset does not exist at the given path or if new path is to a different directory.
        """
        old_path = convert_pathlib_to_string_if_needed(old_path)
        new_path = convert_pathlib_to_string_if_needed(new_path)

        if creds is None:
            creds = {}

        feature_report_path(old_path, "rename", {}, token=token)

        ds = deeplake.load(old_path, verbose=False, token=token, creds=creds)
        ds.rename(new_path)

        return ds  # type: ignore

    @staticmethod
    @spinner
    def delete(
        path: Union[str, pathlib.Path],
        force: bool = False,
        large_ok: bool = False,
        creds: Optional[Union[dict, str]] = None,
        token: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        """Deletes a dataset at a given path.

        Args:
            path (str, pathlib.Path): The path to the dataset to be deleted.
            force (bool): Delete data regardless of whether
                it looks like a deeplake dataset. All data at the path will be removed if set to ``True``.
            large_ok (bool): Delete datasets larger than 1GB. Disabled by default.
            creds (dict, str, optional): The string ``ENV`` or a dictionary containing credentials used to access the dataset at the path.
                - If 'aws_access_key_id', 'aws_secret_access_key', 'aws_session_token' are present, these take precedence over credentials present in the environment or in credentials file. Currently only works with s3 paths.
                - It supports 'aws_access_key_id', 'aws_secret_access_key', 'aws_session_token', 'endpoint_url', 'aws_region', 'profile_name' as keys.
                - If 'ENV' is passed, credentials are fetched from the environment variables. This is also the case when creds is not passed for cloud datasets. For datasets connected to hub cloud, specifying 'ENV' will override the credentials fetched from Activeloop and use local ones.
            token (str, optional): Activeloop token, used for fetching credentials to the dataset at path if it is a Deep Lake dataset. This is optional, tokens are normally autogenerated.
            verbose (bool): If True, logs will be printed. Defaults to True.

        Raises:
            DatasetHandlerError: If a Dataset does not exist at the given path and ``force = False``.
            UserNotLoggedInException: When user is not logged in.
            NotImplementedError: When attempting to delete a managed view.
            ValueError: If version is specified in the path

        Warning:
            This is an irreversible operation. Data once deleted cannot be recovered.
        """
        path, address = process_dataset_path(path)

        if address:
            raise ValueError(
                "deeplake.delete does not accept version address in the dataset path."
            )

        if creds is None:
            creds = {}

        feature_report_path(
            path, "delete", {"Force": force, "Large_OK": large_ok}, token=token
        )

        try:
            qtokens = ["/.queries/", "\\.queries\\"]
            for qt in qtokens:
                if qt in path:
                    raise NotImplementedError(
                        "Deleting managed views by path is not supported. Load the source dataset and do `ds.delete_view(id)` instead."
                    )
            try:
                ds = deeplake.load(path, verbose=False, token=token, creds=creds)
            except UserNotLoggedInException:
                raise UserNotLoggedInException from None
            ds.delete(large_ok=large_ok)
            if verbose:
                logger.info(f"{path} dataset deleted successfully.")
        except Exception as e:
            if force:
                base_storage = storage_provider_from_path(
                    path=path,
                    creds=creds,
                    read_only=False,
                    token=token,
                )
                if len(base_storage) == 0:
                    raise DatasetHandlerError(
                        f"Path {path} is empty or does not exist. Cannot delete."
                    )

                try:
                    base_storage.clear()
                except Exception as e2:
                    raise DatasetHandlerError(
                        "Dataset delete failed. See traceback for more information."
                    ) from e2

                remove_path_from_backend(path, token)
                if verbose:
                    logger.info(f"{path} folder deleted successfully.")
            else:
                if isinstance(e, (DatasetHandlerError, PathNotEmptyException)):
                    raise DatasetHandlerError(
                        "A Deep Lake dataset wasn't found at the specified path. "
                        "This may be due to a corrupt dataset or a wrong path. "
                        "If you want to delete the data at the path regardless, use force=True"
                    )
                raise

    @staticmethod
    @spinner
    def like(
        dest: Union[str, pathlib.Path],
        src: Union[str, Dataset, pathlib.Path],
        runtime: Optional[Dict] = None,
        tensors: Optional[List[str]] = None,
        overwrite: bool = False,
        creds: Optional[Union[dict, str]] = None,
        token: Optional[str] = None,
        org_id: Optional[str] = None,
        public: bool = False,
        verbose: bool = True,
    ) -> Dataset:
        """Creates a new dataset by copying the ``source`` dataset's structure to a new location. No samples are copied,
        only the meta/info for the dataset and it's tensors.

        Args:
            dest: Empty Dataset or Path where the new dataset will be created.
            src (Union[str, Dataset]): Path or dataset object that will be used as the template for the new dataset.
            runtime (dict): Parameters for Activeloop DB Engine. Only applicable for hub:// paths.
            tensors (List[str], optional): Names of tensors (and groups) to be replicated. If not specified all tensors in source dataset are considered.
            overwrite (bool): If True and a dataset exists at `destination`, it will be overwritten. Defaults to False.
            creds (dict, str, optional): The string ``ENV`` or a dictionary containing credentials used to access the dataset at the path.
                - If 'aws_access_key_id', 'aws_secret_access_key', 'aws_session_token' are present, these take precedence over credentials present in the environment or in credentials file. Currently only works with s3 paths.
                - It supports 'aws_access_key_id', 'aws_secret_access_key', 'aws_session_token', 'endpoint_url', 'aws_region', 'profile_name' as keys.
                - If 'ENV' is passed, credentials are fetched from the environment variables. This is also the case when creds is not passed for cloud datasets. For datasets connected to hub cloud, specifying 'ENV' will override the credentials fetched from Activeloop and use local ones.
            token (str, optional): Activeloop token, used for fetching credentials to the dataset at path if it is a Deep Lake dataset. This is optional, tokens are normally autogenerated.
            org_id (str, Optional): Organization id to be used for enabling high-performance features. Only applicable for local datasets.
            public (bool): Defines if the dataset will have public access. Applicable only if Deep Lake cloud storage is used and a new Dataset is being created. Defaults to False.
            verbose (bool): If True, logs will be printed. Defaults to ``True``.


        Returns:
            Dataset: New dataset object.

        Raises:
            ValueError: If ``org_id`` is specified for a non-local dataset.
        """
        if isinstance(dest, Dataset):
            path = dest.path
        else:
            path = dest

        if org_id is not None and get_path_type(path) != "local":
            raise ValueError("org_id parameter can only be used with local datasets")

        feature_report_path(
            path,
            "like",
            {"Overwrite": overwrite, "Public": public, "Tensors": tensors},
            token=token,
        )
        return dataset._like(
            dest,
            src,
            runtime,
            tensors,
            overwrite,
            creds,
            token,
            org_id,
            public,
            verbose,
        )

    @staticmethod
    def _like(  # (No reporting)
        dest,
        src: Union[str, Dataset],
        runtime: Optional[Dict] = None,
        tensors: Optional[List[str]] = None,
        overwrite: bool = False,
        creds: Optional[Union[dict, str]] = None,
        token: Optional[str] = None,
        org_id: Optional[str] = None,
        public: bool = False,
        verbose: bool = True,
        unlink: Union[List[str], bool] = False,
    ) -> Dataset:
        """Copies the `source` dataset's structure to a new location. No samples are copied, only the meta/info for the dataset and it's tensors.

        Args:
            dest: Empty Dataset or Path where the new dataset will be created.
            src (Union[str, Dataset]): Path or dataset object that will be used as the template for the new dataset.
            runtime (dict): Parameters for Activeloop DB Engine. Only applicable for hub:// paths.
            tensors (List[str], optional): Names of tensors (and groups) to be replicated. If not specified all tensors in source dataset are considered.
            dest (str, pathlib.Path, Dataset): Empty Dataset or Path where the new dataset will be created.
            src (Union[str, pathlib.Path, Dataset]): Path or dataset object that will be used as the template for the new dataset.
            overwrite (bool): If True and a dataset exists at `destination`, it will be overwritten. Defaults to False.
            creds (dict, str, optional): The string ``ENV`` or a dictionary containing credentials used to access the dataset at the path.
                - If 'aws_access_key_id', 'aws_secret_access_key', 'aws_session_token' are present, these take precedence over credentials present in the environment or in credentials file. Currently only works with s3 paths.
                - It supports 'aws_access_key_id', 'aws_secret_access_key', 'aws_session_token', 'endpoint_url', 'aws_region', 'profile_name' as keys.
                - If 'ENV' is passed, credentials are fetched from the environment variables. This is also the case when creds is not passed for cloud datasets. For datasets connected to hub cloud, specifying 'ENV' will override the credentials fetched from Activeloop and use local ones.
            token (str, optional): Activeloop token, used for fetching credentials to the dataset at path if it is a Deep Lake dataset. This is optional, tokens are normally autogenerated.
            org_id (str, Optional): Organization id to be used for enabling high-performance features. Only applicable for local datasets.
            public (bool): Defines if the dataset will have public access. Applicable only if Deep Lake cloud storage is used and a new Dataset is being created. Defaults to ``False``.
            verbose (bool): If True, logs will be printed. Defaults to ``True``.
            unlink (Union[List[str], bool]): List of tensors to be unlinked. If ``True`` passed all tensors will be unlinked. Defaults to ``False``, no tensors are unlinked.

        Returns:
            Dataset: New dataset object.
        """

        src = convert_pathlib_to_string_if_needed(src)
        if isinstance(src, str):
            source_ds = dataset.load(src, verbose=verbose)
            src_path = src
        else:
            source_ds = src
            src_path = src.path

        if tensors:
            tensors = source_ds._resolve_tensor_list(tensors)  # type: ignore
        else:
            tensors = source_ds.tensors  # type: ignore

        dest = convert_pathlib_to_string_if_needed(dest)
        if isinstance(dest, Dataset):
            destination_ds = dest
            dest_path = dest.path
        else:
            dest_path = dest
            common_kwargs = {
                "creds": creds,
                "token": token,
                "org_id": org_id,
                "verbose": verbose,
            }
            if dest_path == src_path:
                destination_ds = dataset.load(
                    dest_path, read_only=False, **common_kwargs
                )
            else:
                destination_ds = dataset.empty(
                    dest_path,
                    runtime=runtime,
                    public=public,
                    overwrite=overwrite,
                    **common_kwargs,  # type: ignore
                )

        feature_report_path(
            dest_path, "like", {"Overwrite": overwrite, "Public": public}, token=token
        )

        if unlink is True:
            unlink = tensors  # type: ignore
        elif unlink is False:
            unlink = []
        for tensor_name in tensors:  # type: ignore
            source_tensor = source_ds[tensor_name]
            if overwrite and tensor_name in destination_ds:
                if dest_path == src_path:
                    # load tensor data to memory before deleting
                    # in case of in-place deeplake.like
                    meta = source_tensor.meta
                    info = source_tensor.info
                    sample_shape_tensor = source_tensor._sample_shape_tensor
                    sample_id_tensor = source_tensor._sample_id_tensor
                    sample_info_tensor = source_tensor._sample_info_tensor
                destination_ds.delete_tensor(tensor_name)
            destination_ds.create_tensor_like(tensor_name, source_tensor, unlink=tensor_name in unlink)  # type: ignore

        destination_ds.info.update(source_ds.info.__getstate__())  # type: ignore

        return destination_ds

    @staticmethod
    def copy(
        src: Union[str, pathlib.Path, Dataset],
        dest: Union[str, pathlib.Path],
        runtime: Optional[dict] = None,
        tensors: Optional[List[str]] = None,
        overwrite: bool = False,
        src_creds=None,
        dest_creds=None,
        token=None,
        num_workers: int = 0,
        scheduler="threaded",
        progressbar=True,
        **kwargs,
    ):
        """Copies dataset at ``src`` to ``dest``. Version control history is not included.

        Args:
            src (str, Dataset, pathlib.Path): The Dataset or the path to the dataset to be copied.
            dest (str, pathlib.Path): Destination path to copy to.
            runtime (dict): Parameters for Activeloop DB Engine. Only applicable for hub:// paths.
            tensors (List[str], optional): Names of tensors (and groups) to be copied. If not specified all tensors are copied.
            overwrite (bool): If True and a dataset exists at ``dest``, it will be overwritten. Defaults to ``False``.
            src_creds (dict, str, optional): The string ``ENV`` or a dictionary containing credentials used to access the dataset at the path.
                - If 'aws_access_key_id', 'aws_secret_access_key', 'aws_session_token' are present, these take precedence over credentials present in the environment or in credentials file. Currently only works with s3 paths.
                - It supports 'aws_access_key_id', 'aws_secret_access_key', 'aws_session_token', 'endpoint_url', 'aws_region', 'profile_name' as keys.
                - If 'ENV' is passed, credentials are fetched from the environment variables. This is also the case when creds is not passed for cloud datasets. For datasets connected to hub cloud, specifying 'ENV' will override the credentials fetched from Activeloop and use local ones.
            dest_creds (dict, optional): creds required to create / overwrite datasets at ``dest``.
            token (str, optional): Activeloop token, used for fetching credentials to the dataset at path if it is a Deep Lake dataset. This is optional, tokens are normally autogenerated.
            num_workers (int): The number of workers to use for copying. Defaults to 0. When set to 0, it will always use serial processing, irrespective of the scheduler.
            scheduler (str): The scheduler to be used for copying. Supported values include: 'serial', 'threaded', 'processed' and 'ray'.
                Defaults to 'threaded'.
            progressbar (bool): Displays a progress bar if True (default).
            **kwargs (dict): Additional keyword arguments

        Returns:
            Dataset: New dataset object.

        Raises:
            DatasetHandlerError: If a dataset already exists at destination path and overwrite is False.
            UnsupportedParameterException: If a parameter that is no longer supported is specified.
            DatasetCorruptError: If loading source dataset fails with DatasetCorruptedError.
        """
        if "src_token" in kwargs:
            raise UnsupportedParameterException(
                "src_token is now not supported. You should use `token` instead."
            )

        if "dest_token" in kwargs:
            raise UnsupportedParameterException(
                "dest_token is now not supported. You should use `token` instead."
            )

        if isinstance(src, (str, pathlib.Path)):
            src = convert_pathlib_to_string_if_needed(src)
            try:
                src_ds = deeplake.load(
                    src, read_only=True, creds=src_creds, token=token, verbose=False
                )
            except DatasetCorruptError as e:
                raise DatasetCorruptError(
                    "The source dataset is corrupted.",
                    "You can try to fix this by loading the dataset with `reset=True` "
                    "which will attempt to reset uncommitted HEAD changes and load the previous version.",
                    e.__cause__,
                )
        else:
            src_ds = src
            src_ds.path = str(src_ds.path)

        dest = convert_pathlib_to_string_if_needed(dest)

        return src_ds.copy(
            dest,
            runtime=runtime,
            tensors=tensors,
            overwrite=overwrite,
            creds=dest_creds,
            token=token,
            num_workers=num_workers,
            scheduler=scheduler,
            progressbar=progressbar,
        )

    @staticmethod
    def deepcopy(
        src: Union[str, pathlib.Path, Dataset],
        dest: Union[str, pathlib.Path],
        runtime: Optional[Dict] = None,
        tensors: Optional[List[str]] = None,
        overwrite: bool = False,
        src_creds=None,
        dest_creds=None,
        token=None,
        num_workers: int = 0,
        scheduler="threaded",
        progressbar=True,
        public: bool = False,
        verbose: bool = True,
        **kwargs,
    ):
        """Copies dataset at ``src`` to ``dest`` including version control history.

        Args:
            src (str, pathlib.Path, Dataset): The Dataset or the path to the dataset to be copied.
            dest (str, pathlib.Path): Destination path to copy to.
            runtime (dict): Parameters for Activeloop DB Engine. Only applicable for hub:// paths.
            tensors (List[str], optional): Names of tensors (and groups) to be copied. If not specified all tensors are copied.
            overwrite (bool): If True and a dataset exists at `destination`, it will be overwritten. Defaults to False.
            src_creds (dict, str, optional): The string ``ENV`` or a dictionary containing credentials used to access the dataset at the path.
                - If 'aws_access_key_id', 'aws_secret_access_key', 'aws_session_token' are present, these take precedence over credentials present in the environment or in credentials file. Currently only works with s3 paths.
                - It supports 'aws_access_key_id', 'aws_secret_access_key', 'aws_session_token', 'endpoint_url', 'aws_region', 'profile_name' as keys.
                - If 'ENV' is passed, credentials are fetched from the environment variables. This is also the case when creds is not passed for cloud datasets. For datasets connected to hub cloud, specifying 'ENV' will override the credentials fetched from Activeloop and use local ones.
            dest_creds (dict, optional): creds required to create / overwrite datasets at ``dest``.
            token (str, optional): Activeloop token, used for fetching credentials to the dataset at path if it is a Deep Lake dataset. This is optional, tokens are normally autogenerated.
            num_workers (int): The number of workers to use for copying. Defaults to 0. When set to 0, it will always use serial processing, irrespective of the scheduler.
            scheduler (str): The scheduler to be used for copying. Supported values include: 'serial', 'threaded', 'processed' and 'ray'.
                Defaults to 'threaded'.
            progressbar (bool): Displays a progress bar if True (default).
            public (bool): Defines if the dataset will have public access. Applicable only if Deep Lake cloud storage is used and a new Dataset is being created. Defaults to ``False``.
            verbose (bool): If True, logs will be printed. Defaults to ``True``.
            **kwargs: Additional keyword arguments

        Returns:
            Dataset: New dataset object.

        Raises:
            DatasetHandlerError: If a dataset already exists at destination path and overwrite is False.
            TypeError: If source is not a dataset.
            UnsupportedParameterException: If parameter that is no longer supported is beeing called.
            DatasetCorruptError: If loading source dataset fails with DatasetCorruptedError
        """

        if "src_token" in kwargs:
            raise UnsupportedParameterException(
                "src_token is now not supported. You should use `token` instead."
            )

        if "dest_token" in kwargs:
            raise UnsupportedParameterException(
                "dest_token is now not supported. You should use `token` instead."
            )

        deeplake_reporter.feature_report(
            feature_name="deepcopy",
            parameters={
                "tensors": tensors,
                "overwrite": overwrite,
                "num_workers": num_workers,
                "scheduler": scheduler,
                "progressbar": progressbar,
                "public": public,
                "verbose": verbose,
            },
        )

        dest = convert_pathlib_to_string_if_needed(dest)

        if isinstance(src, (str, pathlib.Path)):
            src = convert_pathlib_to_string_if_needed(src)
            try:
                src_ds = deeplake.load(
                    src, read_only=True, creds=src_creds, token=token, verbose=False
                )
            except DatasetCorruptError as e:
                raise DatasetCorruptError(
                    "The source dataset is corrupted.",
                    "You can try to fix this by loading the dataset with `reset=True` "
                    "which will attempt to reset uncommitted HEAD changes and load the previous version.",
                    e.__cause__,
                )
        else:
            if not isinstance(src, Dataset):
                raise TypeError(
                    "The specified ``src`` is not an allowed type. Please specify a dataset or a materialized dataset view."
                )

            if not src.index.is_trivial():
                raise TypeError(
                    "Deepcopy is not supported for unmaterialized dataset views, i.e. slices of datasets. Please specify a dataset or a materialized dataset view."
                )

            if not src._is_root():
                raise TypeError(
                    "Deepcopy is not supported for individual groups. Please specify a dataset."
                )

            src_ds = src

        verify_dataset_name(dest)

        src_storage = get_base_storage(src_ds.storage)

        db_engine = parse_runtime_parameters(dest, runtime)["tensor_db"]
        dest_storage, cache_chain = get_storage_and_cache_chain(
            dest,
            db_engine=db_engine,
            creds=dest_creds,
            token=token,
            read_only=False,
            memory_cache_size=DEFAULT_MEMORY_CACHE_SIZE,
            local_cache_size=DEFAULT_LOCAL_CACHE_SIZE,
        )

        if dataset_exists(cache_chain):
            if overwrite:
                try:
                    cache_chain.clear()
                except Exception as e:
                    raise DatasetHandlerError(
                        "Dataset overwrite failed. See traceback for more information."
                    ) from e
            else:
                raise DatasetHandlerError(
                    f"A dataset already exists at the given path ({dest}). If you want to copy to a new dataset, either specify another path or use overwrite=True."
                )

        metas: Dict[str, DatasetMeta] = {}

        def copy_func(keys, progress_callback=None):
            cache = generate_chain(
                src_storage,
                memory_cache_size=DEFAULT_MEMORY_CACHE_SIZE,
                local_cache_size=DEFAULT_LOCAL_CACHE_SIZE,
                path=src_ds.path,
            )
            for key in keys:
                # don't copy the lock file
                if key == DATASET_LOCK_FILENAME:
                    continue
                val = metas.get(key) or cache[key]
                if isinstance(val, DeepLakeMemoryObject):
                    dest_storage[key] = val.tobytes()
                else:
                    dest_storage[key] = val
                if progress_callback:
                    progress_callback(1)

        def copy_func_with_progress_bar(pg_callback, keys):
            copy_func(keys, pg_callback)

        keys = src_storage._all_keys()
        if tensors is not None:
            required_tensors = src_ds._resolve_tensor_list(tensors)
            for t in required_tensors[:]:
                required_tensors.extend(src_ds[t].meta.links)
            required_tensor_paths = set(
                src_ds.meta.tensor_names[t] for t in required_tensors
            )

            all_tensors_in_src = src_ds._tensors()
            all_tensor_paths_in_src = [
                src_ds.meta.tensor_names[t] for t in all_tensors_in_src
            ]
            tensor_paths_to_exclude = [
                t for t in all_tensor_paths_in_src if t not in required_tensor_paths
            ]

            def fltr(k):
                for t in tensor_paths_to_exclude:
                    if k.startswith(t + "/") or "/" + t + "/" in k:
                        return False
                return True

            def keep_group(g):
                for t in tensors:
                    if t == g or t.startswith(g + "/"):
                        return True
                return False

            def process_meta(k):
                if posixpath.basename(k) == DATASET_META_FILENAME:
                    meta = DatasetMeta.frombuffer(src_storage[k])
                    if not meta.tensor_names:  # backward compatibility
                        meta.tensor_names = {t: t for t in meta.tensors}
                    meta.tensors = list(
                        filter(
                            lambda t: meta.tensor_names[t] in required_tensor_paths,
                            meta.tensors,
                        )
                    )
                    meta.hidden_tensors = list(
                        filter(lambda t: t in meta.tensors, meta.hidden_tensors)
                    )
                    meta.groups = list(filter(keep_group, meta.groups))
                    meta.tensor_names = {
                        k: v for k, v in meta.tensor_names.items() if k in meta.tensors
                    }
                    metas[k] = meta
                return k

            keys = filter(fltr, map(process_meta, keys))
        keys = list(keys)
        if tensors:
            assert metas
        len_keys = len(keys)
        if num_workers <= 1:
            keys = [keys]
        else:
            keys = [keys[i::num_workers] for i in range(num_workers)]
        compute_provider = get_compute_provider(scheduler, num_workers)
        try:
            if progressbar:
                compute_provider.map_with_progress_bar(
                    copy_func_with_progress_bar,
                    keys,
                    len_keys,
                    "Copying dataset",
                )
            else:
                compute_provider.map(copy_func, keys)
        finally:
            compute_provider.close()

        ret = dataset_factory(
            path=dest,
            storage=cache_chain,
            public=public,
            token=token,
            verbose=verbose,
        )
        ret._register_dataset()
        dataset_created(ret)
        dataset_written(ret)
        if not ret.has_head_changes:
            dataset_committed(ret)
        return ret

    @staticmethod
    @spinner
    def connect(
        src_path: str,
        creds_key: str,
        dest_path: Optional[str] = None,
        org_id: Optional[str] = None,
        ds_name: Optional[str] = None,
        token: Optional[str] = None,
    ) -> Dataset:
        """Connects dataset at ``src_path`` to Deep Lake via the provided path.

        Examples:
            >>> # Connect an s3 dataset
            >>> ds = deeplake.connect(src_path="s3://bucket/dataset", dest_path="hub://my_org/dataset", creds_key="my_managed_credentials_key", token="my_activeloop_token")
            >>> # or
            >>> ds = deeplake.connect(src_path="s3://bucket/dataset", org_id="my_org", creds_key="my_managed_credentials_key", token="my_activeloop_token")

        Args:
            src_path (str): Cloud path to the source dataset. Can be:
                an s3 path like ``s3://bucket/path/to/dataset``.
                a gcs path like ``gcs://bucket/path/to/dataset``.
                an azure path like ``az://account_name/container/path/to/dataset``.
            creds_key (str): The managed credentials to be used for accessing the source path.
            dest_path (str, optional): The full path to where the connected Deep Lake dataset will reside. Can be:
                a Deep Lake path like ``hub://organization/dataset``
            org_id (str, optional): The organization to where the connected Deep Lake dataset will be added.
            ds_name (str, optional): The name of the connected Deep Lake dataset. Will be infered from ``dest_path`` or ``src_path`` if not provided.
            token (str, optional): Activeloop token used to fetch the managed credentials.

        Returns:
            Dataset: The connected Deep Lake dataset.

        Raises:
            InvalidSourcePathError: If the ``src_path`` is not a valid s3, gcs or azure path.
            InvalidDestinationPathError: If ``dest_path``, or ``org_id`` and ``ds_name`` do not form a valid Deep Lake path.
            TokenPermissionError: If the user does not have permission to create a dataset in the specified organization.
        """
        try:
            path = connect_dataset_entry(
                src_path=src_path,
                creds_key=creds_key,
                dest_path=dest_path,
                org_id=org_id,
                ds_name=ds_name,
                token=token,
            )
        except BadRequestException:
            check_param = "organization id" if org_id else "dataset path"
            raise TokenPermissionError(
                "You do not have permission to create a dataset in the specified "
                + check_param
                + "."
                + " Please check the "
                + check_param
                + " and make sure"
                + "that you have sufficient permissions to the organization."
            )
        return deeplake.dataset(path, token=token, verbose=False)

    @staticmethod
    def ingest_coco(
        images_directory: Union[str, pathlib.Path],
        annotation_files: Union[str, pathlib.Path, List[str]],
        dest: Union[str, pathlib.Path],
        key_to_tensor_mapping: Optional[Dict] = None,
        file_to_group_mapping: Optional[Dict] = None,
        ignore_one_group: bool = True,
        ignore_keys: Optional[List[str]] = None,
        image_params: Optional[Dict] = None,
        image_creds_key: Optional[str] = None,
        src_creds: Optional[Union[str, Dict]] = None,
        dest_creds: Optional[Union[str, Dict]] = None,
        inspect_limit: int = 1000000,
        progressbar: bool = True,
        shuffle: bool = False,
        num_workers: int = 0,
        token: Optional[str] = None,
        connect_kwargs: Optional[Dict] = None,
        **dataset_kwargs,
    ) -> Dataset:
        """Ingest images and annotations in COCO format to a Deep Lake Dataset. The source data can be stored locally or in the cloud.

        Examples:
            >>> # Ingest local data in COCO format to a Deep Lake dataset stored in Deep Lake storage.
            >>> ds = deeplake.ingest_coco(
            >>>     "<path/to/images/directory>",
            >>>     ["path/to/annotation/file1.json", "path/to/annotation/file2.json"],
            >>>     dest="hub://org_id/dataset",
            >>>     key_to_tensor_mapping={"category_id": "labels", "bbox": "boxes"},
            >>>     file_to_group_mapping={"file1.json": "group1", "file2.json": "group2"},
            >>>     ignore_keys=["area", "image_id", "id"],
            >>>     num_workers=4,
            >>> )
            >>> # Ingest data from your cloud into another Deep Lake dataset in your cloud, and connect that dataset to the Deep Lake backend.
            >>> ds = deeplake.ingest_coco(
            >>>     "s3://bucket/images/directory",
            >>>     "s3://bucket/annotation/file1.json",
            >>>     dest="s3://bucket/dataset_name",
            >>>     ignore_one_group=True,
            >>>     ignore_keys=["area", "image_id", "id"],
            >>>     image_settings={"name": "images", "htype": "link[image]", "sample_compression": "jpeg"},
            >>>     image_creds_key="my_s3_managed_credentials",
            >>>     src_creds=aws_creds, # Can also be inferred from environment
            >>>     dest_creds=aws_creds, # Can also be inferred from environment
            >>>     connect_kwargs={"creds_key": "my_s3_managed_credentials", "org_id": "org_id"},
            >>>     num_workers=4,
            >>> )

        Args:
            images_directory (str, pathlib.Path): The path to the directory containing images.
            annotation_files (str, pathlib.Path, List[str]): Path to JSON annotation files in COCO format.
            dest (str, pathlib.Path):
                - The full path to the dataset. Can be:
                - a Deep Lake cloud path of the form ``hub://org_id/datasetname``. To write to Deep Lake cloud datasets, ensure that you are logged in to Deep Lake (use 'activeloop login' from command line), or pass in a token using the 'token' parameter.
                - an s3 path of the form ``s3://bucketname/path/to/dataset``. Credentials are required in either the environment or passed to the creds argument.
                - a local file system path of the form ``./path/to/dataset`` or ``~/path/to/dataset`` or ``path/to/dataset``.
                - a memory path of the form ``mem://path/to/dataset`` which doesn't save the dataset but keeps it in memory instead. Should be used only for testing as it does not persist.
            key_to_tensor_mapping (Optional[Dict]): A one-to-one mapping between COCO keys and Dataset tensor names.
            file_to_group_mapping (Optional[Dict]): A one-to-one mapping between COCO annotation file names and Dataset group names.
            ignore_one_group (bool): Skip creation of group in case of a single annotation file. Set to ``False`` by default.
            ignore_keys (List[str]): A list of COCO keys to ignore.
            image_params (Optional[Dict]): A dictionary containing parameters for the images tensor.
            image_creds_key (Optional[str]): The name of the managed credentials to use for accessing the images in the linked tensor (is applicable).
            src_creds (Optional[Union[str, Dict]]): Credentials to access the source data. If not provided, will be inferred from the environment.
            dest_creds (Optional[Union[str, Dict]]): The string ``ENV`` or a dictionary containing credentials used to access the destination path of the dataset.
            inspect_limit (int): The maximum number of samples to inspect in the annotations json, in order to generate the set of COCO annotation keys. Set to ``1000000`` by default.
            progressbar (bool): Enables or disables ingestion progress bar. Set to ``True`` by default.
            shuffle (bool): Shuffles the input data prior to ingestion. Set to ``False`` by default.
            num_workers (int): The number of workers to use for ingestion. Set to ``0`` by default.
            token (Optional[str]): The token to use for accessing the dataset and/or connecting it to Deep Lake.
            connect_kwargs (Optional[Dict]): If specified, the dataset will be connected to Deep Lake, and connect_kwargs will be passed to :meth:`Dataset.connect <deeplake.core.dataset.Dataset.connect>`.
            **dataset_kwargs: Any arguments passed here will be forwarded to the dataset creator function. See :func:`deeplake.empty`.

        Returns:
            Dataset: The Dataset created from images and COCO annotations.

        Raises:
            IngestionError: If either ``key_to_tensor_mapping`` or ``file_to_group_mapping`` are not one-to-one.
        """

        dest = convert_pathlib_to_string_if_needed(dest)
        images_directory = convert_pathlib_to_string_if_needed(images_directory)
        annotation_files = (
            [convert_pathlib_to_string_if_needed(f) for f in annotation_files]
            if isinstance(annotation_files, list)
            else convert_pathlib_to_string_if_needed(annotation_files)
        )

        feature_report_path(
            dest,
            "ingest_coco",
            {"num_workers": num_workers},
            token=token,
        )

        unstructured = CocoDataset(
            source=images_directory,
            annotation_files=annotation_files,
            key_to_tensor_mapping=key_to_tensor_mapping,
            file_to_group_mapping=file_to_group_mapping,
            ignore_one_group=ignore_one_group,
            ignore_keys=ignore_keys,
            image_params=image_params,
            image_creds_key=image_creds_key,
            creds=src_creds,
        )
        structure = unstructured.prepare_structure(inspect_limit)

        ds = deeplake.empty(
            dest, creds=dest_creds, verbose=False, token=token, **dataset_kwargs
        )
        if connect_kwargs is not None:
            connect_kwargs["token"] = token or connect_kwargs.get("token")
            ds.connect(**connect_kwargs)

        structure.create_missing(ds)

        unstructured.structure(ds, progressbar, num_workers, shuffle)

        return ds

    @staticmethod
    def ingest_yolo(
        data_directory: Union[str, pathlib.Path],
        dest: Union[str, pathlib.Path],
        class_names_file: Optional[Union[str, pathlib.Path]] = None,
        annotations_directory: Optional[Union[str, pathlib.Path]] = None,
        allow_no_annotation: bool = False,
        image_params: Optional[Dict] = None,
        label_params: Optional[Dict] = None,
        coordinates_params: Optional[Dict] = None,
        src_creds: Optional[Union[str, Dict]] = None,
        dest_creds: Optional[Union[str, Dict]] = None,
        image_creds_key: Optional[str] = None,
        inspect_limit: int = 1000,
        progressbar: bool = True,
        shuffle: bool = False,
        num_workers: int = 0,
        token: Optional[str] = None,
        connect_kwargs: Optional[Dict] = None,
        **dataset_kwargs,
    ) -> Dataset:
        """Ingest images and annotations (bounding boxes or polygons) in YOLO format to a Deep Lake Dataset. The source data can be stored locally or in the cloud.

        Examples:
            >>> # Ingest local data in YOLO format to a Deep Lake dataset stored in Deep Lake storage.
            >>> ds = deeplake.ingest_yolo(
            >>>     "path/to/data/directory",
            >>>     dest="hub://org_id/dataset",
            >>>     allow_no_annotation=True,
            >>>     token="my_activeloop_token",
            >>>     num_workers=4,
            >>> )
            >>> # Ingest data from your cloud into another Deep Lake dataset in your cloud, and connect that dataset to the Deep Lake backend.
            >>> ds = deeplake.ingest_yolo(
            >>>     "s3://bucket/data_directory",
            >>>     dest="s3://bucket/dataset_name",
            >>>     image_params={"name": "image_links", "htype": "link[image]"},
            >>>     image_creds_key="my_s3_managed_credentials",
            >>>     src_creds=aws_creds, # Can also be inferred from environment
            >>>     dest_creds=aws_creds, # Can also be inferred from environment
            >>>     connect_kwargs={"creds_key": "my_s3_managed_credentials", "org_id": "org_id"},
            >>>     num_workers=4,
            >>> )

        Args:
            data_directory (str, pathlib.Path): The path to the directory containing the data (images files and annotation files(see 'annotations_directory' input for specifying annotations in a separate directory).
            dest (str, pathlib.Path):
                - The full path to the dataset. Can be:
                - a Deep Lake cloud path of the form ``hub://org_id/datasetname``. To write to Deep Lake cloud datasets, ensure that you are logged in to Deep Lake (use 'activeloop login' from command line), or pass in a token using the 'token' parameter.
                - an s3 path of the form ``s3://bucketname/path/to/dataset``. Credentials are required in either the environment or passed to the creds argument.
                - a local file system path of the form ``./path/to/dataset`` or ``~/path/to/dataset`` or ``path/to/dataset``.
                - a memory path of the form ``mem://path/to/dataset`` which doesn't save the dataset but keeps it in memory instead. Should be used only for testing as it does not persist.
            class_names_file: Path to the file containing the class names on separate lines. This is typically a file titled classes.names.
            annotations_directory (Optional[Union[str, pathlib.Path]]): Path to directory containing the annotations. If specified, the 'data_directory' will not be examined for annotations.
            allow_no_annotation (bool): Flag to determine whether missing annotations files corresponding to an image should be treated as empty annoations. Set to ``False`` by default.
            image_params (Optional[Dict]): A dictionary containing parameters for the images tensor.
            label_params (Optional[Dict]): A dictionary containing parameters for the labels tensor.
            coordinates_params (Optional[Dict]): A dictionary containing parameters for the ccoordinates tensor. This tensor either contains bounding boxes or polygons.
            src_creds (Optional[Union[str, Dict]]): Credentials to access the source data. If not provided, will be inferred from the environment.
            dest_creds (Optional[Union[str, Dict]]): The string ``ENV`` or a dictionary containing credentials used to access the destination path of the dataset.
            image_creds_key (Optional[str]): creds_key for linked tensors, applicable if the htype for the images tensor is specified as 'link[image]' in the 'image_params' input.
            inspect_limit (int): The maximum number of annotations to inspect, in order to infer whether they are bounding boxes of polygons. This in put is ignored if the htype is specfied in the 'coordinates_params'.
            progressbar (bool): Enables or disables ingestion progress bar. Set to ``True`` by default.
            shuffle (bool): Shuffles the input data prior to ingestion. Set to ``False`` by default.
            num_workers (int): The number of workers to use for ingestion. Set to ``0`` by default.
            token (Optional[str]): The token to use for accessing the dataset and/or connecting it to Deep Lake.
            connect_kwargs (Optional[Dict]): If specified, the dataset will be connected to Deep Lake, and connect_kwargs will be passed to :meth:`Dataset.connect <deeplake.core.dataset.Dataset.connect>`.
            **dataset_kwargs: Any arguments passed here will be forwarded to the dataset creator function. See :func:`deeplake.empty`.

        Returns:
            Dataset: The Dataset created from the images and YOLO annotations.

        Raises:
            IngestionError: If annotations are not found for all the images and 'allow_no_annotation' is False
        """

        dest = convert_pathlib_to_string_if_needed(dest)
        data_directory = convert_pathlib_to_string_if_needed(data_directory)

        annotations_directory = (
            convert_pathlib_to_string_if_needed(annotations_directory)
            if annotations_directory is not None
            else None
        )

        class_names_file = (
            convert_pathlib_to_string_if_needed(class_names_file)
            if class_names_file is not None
            else None
        )

        feature_report_path(
            dest,
            "ingest_yolo",
            {"num_workers": num_workers},
            token=token,
        )

        unstructured = YoloDataset(
            data_directory=data_directory,
            class_names_file=class_names_file,
            annotations_directory=annotations_directory,
            image_params=image_params,
            label_params=label_params,
            coordinates_params=coordinates_params,
            allow_no_annotation=allow_no_annotation,
            creds=src_creds,
            image_creds_key=image_creds_key,
            inspect_limit=inspect_limit,
        )

        structure = unstructured.prepare_structure()

        ds = deeplake.empty(
            dest, creds=dest_creds, verbose=False, token=token, **dataset_kwargs
        )
        if connect_kwargs is not None:
            connect_kwargs["token"] = token or connect_kwargs.get("token")
            ds.connect(**connect_kwargs)

        structure.create_missing(ds)

        unstructured.structure(
            ds,
            progressbar,
            num_workers,
            shuffle,
        )

        return ds

    @staticmethod
    def ingest_classification(
        src: Union[str, pathlib.Path],
        dest: Union[str, pathlib.Path],
        image_params: Optional[Dict] = None,
        label_params: Optional[Dict] = None,
        dest_creds: Optional[Union[str, Dict]] = None,
        progressbar: bool = True,
        summary: bool = True,
        num_workers: int = 0,
        shuffle: bool = True,
        token: Optional[str] = None,
        connect_kwargs: Optional[Dict] = None,
        **dataset_kwargs,
    ) -> Dataset:
        """Ingest a dataset of images from a local folder to a Deep Lake Dataset. Images should be stored in subfolders by class name.

        Args:
            src (str, pathlib.Path): Local path to where the unstructured dataset of images is stored or path to csv file.
            dest (str, pathlib.Path): - The full path to the dataset. Can be:
                - a Deep Lake cloud path of the form ``hub://org_id/datasetname``. To write to Deep Lake cloud datasets, ensure that you are logged in to Deep Lake (use 'activeloop login' from command line)
                - an s3 path of the form ``s3://bucketname/path/to/dataset``. Credentials are required in either the environment or passed to the creds argument.
                - a local file system path of the form ``./path/to/dataset`` or ``~/path/to/dataset`` or ``path/to/dataset``.
                - a memory path of the form ``mem://path/to/dataset`` which doesn't save the dataset but keeps it in memory instead. Should be used only for testing as it does not persist.
            image_params (Optional[Dict]): A dictionary containing parameters for the images tensor.
            label_params (Optional[Dict]): A dictionary containing parameters for the labels tensor.
            dest_creds (Optional[Union[str, Dict]]): The string ``ENV`` or a dictionary containing credentials used to access the destination path of the dataset.
            progressbar (bool): Enables or disables ingestion progress bar. Defaults to ``True``.
            summary (bool): If ``True``, a summary of skipped files will be printed after completion. Defaults to ``True``.
            num_workers (int): The number of workers to use for ingestion. Set to ``0`` by default.
            shuffle (bool): Shuffles the input data prior to ingestion. Since data arranged in folders by class is highly non-random, shuffling is important in order to produce optimal results when training. Defaults to ``True``.
            token (Optional[str]): The token to use for accessing the dataset.
            connect_kwargs (Optional[Dict]): If specified, the dataset will be connected to Deep Lake, and connect_kwargs will be passed to :meth:`Dataset.connect <deeplake.core.dataset.Dataset.connect>`.
            **dataset_kwargs: Any arguments passed here will be forwarded to the dataset creator function see :func:`deeplake.empty`.

        Returns:
            Dataset: New dataset object with structured dataset.

        Raises:
            InvalidPathException: If the source directory does not exist.
            SamePathException: If the source and destination path are same.
            AutoCompressionError: If the source director is empty or does not contain a valid extension.
            InvalidFileExtension: If the most frequent file extension is found to be 'None' during auto-compression.

        Note:
            - Currently only local source paths and image classification datasets / csv files are supported for automatic ingestion.
            - Supported filetypes: png/jpeg/jpg/csv.
            - All files and sub-directories with unsupported filetypes are ignored.
            - Valid source directory structures for image classification look like::

                data/
                    img0.jpg
                    img1.jpg
                    ...

            - or::

                data/
                    class0/
                        cat0.jpg
                        ...
                    class1/
                        dog0.jpg
                        ...
                    ...

            - or::

                data/
                    train/
                        class0/
                            img0.jpg
                            ...
                        ...
                    val/
                        class0/
                            img0.jpg
                            ...
                        ...
                    ...

            - Classes defined as sub-directories can be accessed at ``ds["test/labels"].info.class_names``.
            - Support for train and test sub directories is present under ``ds["train/images"]``, ``ds["train/labels"]`` and ``ds["test/images"]``, ``ds["test/labels"]``.
            - Mapping filenames to classes from an external file is currently not supported.
        """
        dest = convert_pathlib_to_string_if_needed(dest)
        feature_report_path(
            dest,
            "ingest_classification",
            {
                "Progressbar": progressbar,
                "Summary": summary,
            },
            token=token,
        )

        src = convert_pathlib_to_string_if_needed(src)

        if isinstance(src, str):
            if os.path.isdir(dest) and os.path.samefile(src, dest):
                raise SamePathException(src)

            if src.lower().endswith((".csv", ".txt")):
                import pandas as pd  # type:ignore

                if not os.path.isfile(src):
                    raise InvalidPathException(src)
                source = pd.read_csv(src, quotechar='"', skipinitialspace=True)
                ds = dataset.ingest_dataframe(
                    source,
                    dest,
                    dest_creds=dest_creds,
                    progressbar=progressbar,
                    token=token,
                    **dataset_kwargs,
                )
                return ds

            if not os.path.isdir(src):
                raise InvalidPathException(src)

            if image_params is None:
                image_params = {}
            if label_params is None:
                label_params = {}

            if not image_params.get("sample_compression", None):
                images_compression = get_most_common_extension(src)
                if images_compression is None:
                    raise InvalidFileExtension(src)
                image_params["sample_compression"] = images_compression

            # TODO: support more than just image classification (and update docstring)
            unstructured = ImageClassification(source=src)

            ds = deeplake.empty(
                dest, creds=dest_creds, token=token, verbose=False, **dataset_kwargs
            )
            if connect_kwargs is not None:
                connect_kwargs["token"] = token or connect_kwargs.get("token")
                ds.connect(**connect_kwargs)

            # TODO: auto detect compression
            unstructured.structure(
                ds,  # type: ignore
                progressbar=progressbar,
                generate_summary=summary,
                image_tensor_args=image_params,
                label_tensor_args=label_params,
                num_workers=num_workers,
                shuffle=shuffle,
            )

        return ds  # type: ignore

    @staticmethod
    def ingest_kaggle(
        tag: str,
        src: Union[str, pathlib.Path],
        dest: Union[str, pathlib.Path],
        exist_ok: bool = False,
        images_compression: str = "auto",
        dest_creds: Optional[Union[str, Dict]] = None,
        kaggle_credentials: Optional[dict] = None,
        progressbar: bool = True,
        summary: bool = True,
        shuffle: bool = True,
        **dataset_kwargs,
    ) -> Dataset:
        """Download and ingest a kaggle dataset and store it as a structured dataset to destination.

        Args:
            tag (str): Kaggle dataset tag. Example: ``"coloradokb/dandelionimages"`` points to https://www.kaggle.com/coloradokb/dandelionimages
            src (str, pathlib.Path): Local path to where the raw kaggle dataset will be downlaoded to.
            dest (str, pathlib.Path): - The full path to the dataset. Can be:
                - a Deep Lake cloud path of the form ``hub://username/datasetname``. To write to Deep Lake cloud datasets, ensure that you are logged in to Deep Lake (use 'activeloop login' from command line)
                - an s3 path of the form ``s3://bucketname/path/to/dataset``. Credentials are required in either the environment or passed to the creds argument.
                - a local file system path of the form ``./path/to/dataset`` or ``~/path/to/dataset`` or ``path/to/dataset``.
                - a memory path of the form ``mem://path/to/dataset`` which doesn't save the dataset but keeps it in memory instead. Should be used only for testing as it does not persist.
            exist_ok (bool): If the kaggle dataset was already downloaded and ``exist_ok`` is ``True``, ingestion will proceed without error.
            images_compression (str): For image classification datasets, this compression will be used for the ``images`` tensor. If ``images_compression`` is "auto", compression will be automatically determined by the most common extension in the directory.
            dest_creds (Optional[Union[str, Dict]]): The string ``ENV`` or a dictionary containing credentials used to access the destination path of the dataset.
            kaggle_credentials (dict): A dictionary containing kaggle credentials {"username":"YOUR_USERNAME", "key": "YOUR_KEY"}. If ``None``, environment variables/the kaggle.json file will be used if available.
            progressbar (bool): Enables or disables ingestion progress bar. Set to ``True`` by default.
            summary (bool): Generates ingestion summary. Set to ``True`` by default.
            shuffle (bool): Shuffles the input data prior to ingestion. Since data arranged in folders by class is highly non-random, shuffling is important in order to produce optimal results when training. Defaults to ``True``.
            **dataset_kwargs: Any arguments passed here will be forwarded to the dataset creator function. See :func:`deeplake.dataset`.

        Returns:
            Dataset: New dataset object with structured dataset.

        Raises:
            SamePathException: If the source and destination path are same.

        Note:
            Currently only local source paths and image classification datasets are supported for automatic ingestion.
        """
        src = convert_pathlib_to_string_if_needed(src)
        dest = convert_pathlib_to_string_if_needed(dest)

        feature_report_path(
            dest,
            "ingest_kaggle",
            {
                "Images_Compression": images_compression,
                "Exist_Ok": exist_ok,
                "Progressbar": progressbar,
                "Summary": summary,
            },
            token=dataset_kwargs.get("token", None),
        )

        if os.path.isdir(src) and os.path.isdir(dest):
            if os.path.samefile(src, dest):
                raise SamePathException(src)

        download_kaggle_dataset(
            tag,
            local_path=src,
            kaggle_credentials=kaggle_credentials,
            exist_ok=exist_ok,
        )

        ds = deeplake.ingest_classification(
            src=src,
            dest=dest,
            image_params={"sample_compression": images_compression},
            dest_creds=dest_creds,
            progressbar=progressbar,
            summary=summary,
            shuffle=shuffle,
            **dataset_kwargs,
        )

        return ds

    @staticmethod
    def ingest_dataframe(
        src,
        dest: Union[str, pathlib.Path],
        column_params: Optional[Dict] = None,
        src_creds: Optional[Union[str, Dict]] = None,
        dest_creds: Optional[Union[str, Dict]] = None,
        creds_key: Optional[Dict] = None,
        progressbar: bool = True,
        token: Optional[str] = None,
        connect_kwargs: Optional[Dict] = None,
        **dataset_kwargs,
    ):
        """Convert pandas dataframe to a Deep Lake Dataset. The contents of the dataframe can be parsed literally, or can be treated as links to local or cloud files.

        Examples:


                    >>> # Ingest local data in COCO format to a Deep Lake dataset stored in Deep Lake storage.
            >>> ds = deeplake.ingest_coco(
            >>>     "<path/to/images/directory>",
            >>>     ["path/to/annotation/file1.json", "path/to/annotation/file2.json"],
            >>>     dest="hub://org_id/dataset",
            >>>     key_to_tensor_mapping={"category_id": "labels", "bbox": "boxes"},
            >>>     file_to_group_mapping={"file1.json": "group1", "file2.json": "group2"},
            >>>     ignore_keys=["area", "image_id", "id"],
            >>>     num_workers=4,
            >>> )
            >>> # Ingest data from your cloud into another Deep Lake dataset in your cloud, and connect that dataset to the Deep Lake backend.



            >>> # Ingest data from a DataFrame into a Deep Lake dataset stored in Deep Lake storage.
            >>> ds = deeplake.ingest_dataframe(
            >>>     df,
            >>>     dest="hub://org_id/dataset",
            >>> )
            >>> # Ingest data from a DataFrame into a Deep Lake dataset stored in Deep Lake storage. The filenames in `df_column_with_cloud_paths` will be used as the filenames for loading data into the dataset.
            >>> ds = deeplake.ingest_dataframe(
            >>>     df,
            >>>     dest="hub://org_id/dataset",
            >>>     column_params={"df_column_with_cloud_paths": {"name": "images", "htype": "image"}},
            >>>     src_creds=aws_creds
            >>> )
            >>> # Ingest data from a DataFrame into a Deep Lake dataset stored in Deep Lake storage. The filenames in `df_column_with_cloud_paths` will be used as the filenames for linked data in the dataset.
            >>> ds = deeplake.ingest_dataframe(
            >>>     df,
            >>>     dest="hub://org_id/dataset",
            >>>     column_params={"df_column_with_cloud_paths": {"name": "image_links", "htype": "link[image]"}},
            >>>     creds_key="my_s3_managed_credentials"
            >>> )
            >>> # Ingest data from a DataFrame into a Deep Lake dataset stored in your cloud, and connect that dataset to the Deep Lake backend. The filenames in `df_column_with_cloud_paths` will be used as the filenames for linked data in the dataset.
            >>> ds = deeplake.ingest_dataframe(
            >>>     df,
            >>>     dest="s3://bucket/dataset_name",
            >>>     column_params={"df_column_with_cloud_paths": {"name": "image_links", "htype": "link[image]"}},
            >>>     creds_key="my_s3_managed_credentials"
            >>>     connect_kwargs={"creds_key": "my_s3_managed_credentials", "org_id": "org_id"},
            >>> )

        Args:
            src (pd.DataFrame): The pandas dataframe to be converted.
            dest (str, pathlib.Path):
                - A Dataset or The full path to the dataset. Can be:
                - a Deep Lake cloud path of the form ``hub://username/datasetname``. To write to Deep Lake cloud datasets, ensure that you are logged in to Deep Lake (use 'activeloop login' from command line)
                - an s3 path of the form ``s3://bucketname/path/to/dataset``. Credentials are required in either the environment or passed to the creds argument.
                - a local file system path of the form ``./path/to/dataset`` or ``~/path/to/dataset`` or ``path/to/dataset``.
                - a memory path of the form ``mem://path/to/dataset`` which doesn't save the dataset but keeps it in memory instead. Should be used only for testing as it does not persist.
            column_params (Optional[Dict]): A dictionary containing parameters for the tensors corresponding to the dataframe columns.
            src_creds (Optional[Union[str, Dict]]): Credentials to access the source data. If not provided, will be inferred from the environment.
            dest_creds (Optional[Union[str, Dict]]): The string ``ENV`` or a dictionary containing credentials used to access the destination path of the dataset.
            creds_key (Optional[str]): creds_key for linked tensors, applicable if the htype any tensor is specified as 'link[...]' in the 'column_params' input.
            progressbar (bool): Enables or disables ingestion progress bar. Set to ``True`` by default.
            token (Optional[str]): The token to use for accessing the dataset.
            connect_kwargs (Optional[Dict]): A dictionary containing arguments to be passed to the dataset connect method. See :meth:`Dataset.connect`.
            **dataset_kwargs: Any arguments passed here will be forwarded to the dataset creator function. See :func:`deeplake.empty`.

        Returns:
            Dataset: New dataset created from the dataframe.

        Raises:
            Exception: If ``src`` is not a valid pandas dataframe object.
        """
        import pandas as pd
        from deeplake.auto.structured.dataframe import DataFrame

        feature_report_path(
            convert_pathlib_to_string_if_needed(dest),
            "ingest_dataframe",
            {},
            token=token,
        )

        if not isinstance(src, pd.DataFrame):
            raise Exception("Source provided is not a valid pandas dataframe object")

        structured = DataFrame(src, column_params, src_creds, creds_key)

        dest = convert_pathlib_to_string_if_needed(dest)
        ds = deeplake.empty(
            dest, creds=dest_creds, token=token, verbose=False, **dataset_kwargs
        )
        if connect_kwargs is not None:
            connect_kwargs["token"] = token or connect_kwargs.get("token")
            ds.connect(**connect_kwargs)

        structured.fill_dataset(ds, progressbar)  # type: ignore

        return ds  # type: ignore

    @staticmethod
    @spinner
    def query(query_string: str, token: Optional[str] = "") -> Dataset:
        from deeplake.enterprise.libdeeplake_query import universal_query

        return universal_query(query_string=query_string, token=token)
