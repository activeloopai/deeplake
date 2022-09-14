import os
import re
import hub
import pathlib
import posixpath
from typing import Dict, Optional, Union, List

from hub.auto.unstructured.kaggle import download_kaggle_dataset
from hub.auto.unstructured.classification import (
    ImageClassification,
    AudioClassification,
    VideoClassification,
)
from hub.client.client import HubBackendClient
from hub.client.log import logger
from hub.core.dataset import Dataset, dataset_factory
from hub.core.meta.dataset_meta import DatasetMeta
from hub.util.path import convert_pathlib_to_string_if_needed
from hub.constants import (
    DEFAULT_MEMORY_CACHE_SIZE,
    DEFAULT_LOCAL_CACHE_SIZE,
    DEFAULT_READONLY,
    DATASET_META_FILENAME,
)
from hub.util.access_method import (
    check_access_method,
    get_local_dataset,
    parse_access_method,
)
from hub.util.auto import get_most_common_extension
from hub.util.bugout_reporter import feature_report_path, hub_reporter
from hub.util.delete_entry import remove_path_from_backend
from hub.util.keys import dataset_exists
from hub.util.exceptions import (
    AgreementError,
    DatasetHandlerError,
    InvalidFileExtension,
    InvalidPathException,
    PathNotEmptyException,
    SamePathException,
    AuthorizationException,
    UserNotLoggedInException,
    TokenPermissionError,
)
from hub.compression import (
    IMAGE_COMPRESSIONS,
    VIDEO_COMPRESSIONS,
    AUDIO_COMPRESSIONS,
    BYTE_COMPRESSIONS,
    COMPRESSION_ALIASES,
)
from hub.util.storage import get_storage_and_cache_chain, storage_provider_from_path
from hub.util.compute import get_compute_provider
from hub.util.remove_cache import get_base_storage
from hub.util.cache_chain import generate_chain
from hub.core.storage.hub_memory_object import HubMemoryObject

_image_compressions = (
    IMAGE_COMPRESSIONS[:] + BYTE_COMPRESSIONS + list(COMPRESSION_ALIASES)
)
_image_compressions.remove("dcm")
_video_compressions = VIDEO_COMPRESSIONS
_audio_compressions = AUDIO_COMPRESSIONS


class dataset:
    @staticmethod
    def init(
        path: Union[str, pathlib.Path],
        read_only: Optional[bool] = None,
        overwrite: bool = False,
        public: bool = False,
        memory_cache_size: int = DEFAULT_MEMORY_CACHE_SIZE,
        local_cache_size: int = DEFAULT_LOCAL_CACHE_SIZE,
        creds: Optional[Union[Dict, str]] = None,
        token: Optional[str] = None,
        verbose: bool = True,
        access_method: str = "stream",
    ):
        """Returns a Dataset object referencing either a new or existing dataset.

        Important:
            Using `overwrite` will delete all of your data if it exists! Be very careful when setting this parameter.

        Examples:
            ```
            ds = hub.dataset("hub://username/dataset")
            ds = hub.dataset("s3://mybucket/my_dataset")
            ds = hub.dataset("./datasets/my_dataset", overwrite=True)
            ```

        Args:
            path (str, pathlib.Path): - The full path to the dataset. Can be:
                - a Hub cloud path of the form `hub://username/datasetname`. To write to Hub cloud datasets, ensure that you are logged in to Hub (use 'activeloop login' from command line)
                - an s3 path of the form `s3://bucketname/path/to/dataset`. Credentials are required in either the environment or passed to the creds argument.
                - a local file system path of the form `./path/to/dataset` or `~/path/to/dataset` or `path/to/dataset`.
                - a memory path of the form `mem://path/to/dataset` which doesn't save the dataset but keeps it in memory instead. Should be used only for testing as it does not persist.
            read_only (bool, optional): Opens dataset in read only mode if this is passed as True. Defaults to False.
                Datasets stored on Hub cloud that your account does not have write access to will automatically open in read mode.
            overwrite (bool): WARNING: If set to True this overwrites the dataset if it already exists. This can NOT be undone! Defaults to False.
            public (bool): Defines if the dataset will have public access. Applicable only if Hub cloud storage is used and a new Dataset is being created. Defaults to True.
            memory_cache_size (int): The size of the memory cache to be used in MB.
            local_cache_size (int): The size of the local filesystem cache to be used in MB.
            creds (dict, optional): - A dictionary containing credentials used to access the dataset at the path.
                - If aws_access_key_id, aws_secret_access_key, aws_session_token are present, these take precedence over credentials present in the environment or in credentials file. Currently only works with s3 paths.
                - It supports 'aws_access_key_id', 'aws_secret_access_key', 'aws_session_token', 'endpoint_url', 'aws_region', 'profile_name' as keys.
            token (str, optional): Activeloop token, used for fetching credentials to the dataset at path if it is a Hub dataset. This is optional, tokens are normally autogenerated.
            verbose (bool): If True, logs will be printed. Defaults to True.
            access_method (str): - The access method to use for the dataset. Can be:
                - 'stream'
                    - Streams the data from the dataset i.e. only fetches data when required. This is the default value.
                - 'download'
                    - Downloads the data to the local filesystem to the path specified in environment variable `HUB_DOWNLOAD_PATH`.
                    - Raises an exception if the environment variable is not set, or if the path is not empty.
                    - Will also raise an exception if the dataset does not exist. The 'download' access method can also be modified to specify num_workers and/or scheduler.
                    - For example: 'download:2:processed', will use 2 workers and use processed scheduler, while 'download:3' will use 3 workers and default scheduler (threaded), and 'download:processed' will use a single worker and use processed scheduler.
                - 'local'
                    - Used when download was already done in a previous run.
                    - Doesn't download the data again.
                    - Raises an exception if `HUB_DOWNLOAD_PATH` environment variable is not set or the dataset is not found in `HUB_DOWNLOAD_PATH`.

                **Note: Any changes made to the dataset in download/local mode will only be made to the local copy and will not be reflected in the original dataset.**

        Returns:
            Dataset object created using the arguments provided.

        Raises:
            AgreementError: When agreement is rejected
            UserNotLoggedInException: When user is not logged in
            InvalidTokenException: If the specified token is invalid
            TokenPermissionError: When there are permission or other errors related to token
        """
        access_method, num_workers, scheduler = parse_access_method(access_method)
        check_access_method(access_method, overwrite)
        path = convert_pathlib_to_string_if_needed(path)

        if creds is None:
            creds = {}

        try:
            storage, cache_chain = get_storage_and_cache_chain(
                path=path,
                read_only=read_only,
                creds=creds,
                token=token,
                memory_cache_size=memory_cache_size,
                local_cache_size=local_cache_size,
            )

            feature_report_path(path, "dataset", {"Overwrite": overwrite}, token=token)
        except Exception as e:
            if isinstance(e, UserNotLoggedInException):
                message = (
                    f"Please log in through the CLI in order to create this dataset, "
                    "or create an API token in the UI and pass it to this method using "
                    "the ‘token’ parameter. The CLI commands are ‘activeloop login’ and "
                    "‘activeloop register."
                )
                raise UserNotLoggedInException(message)
            raise
        ds_exists = dataset_exists(cache_chain)
        if overwrite and ds_exists:
            cache_chain.clear()

        try:
            if access_method == "stream":
                return dataset_factory(
                    path=path,
                    storage=cache_chain,
                    read_only=read_only,
                    public=public,
                    token=token,
                    verbose=verbose,
                )

            return get_local_dataset(
                access_method=access_method,
                path=path,
                read_only=read_only,
                memory_cache_size=memory_cache_size,
                local_cache_size=local_cache_size,
                creds=creds,
                token=token,
                verbose=verbose,
                ds_exists=ds_exists,
                num_workers=num_workers,
                scheduler=scheduler,
            )
        except AgreementError as e:
            raise e from None

    @staticmethod
    def exists(
        path: Union[str, pathlib.Path],
        creds: Optional[dict] = None,
        token: Optional[str] = None,
    ) -> bool:
        """Checks if a dataset exists at the given `path`.
        Args:
            path (str, pathlib.Path): the path which needs to be checked.
            creds (dict, optional): A dictionary containing credentials used to access the dataset at the path.
            token (str, optional): Activeloop token, used for fetching credentials to the dataset at path if it is a Hub dataset. This is optional, tokens are normally autogenerated.
        Returns:
            A boolean confirming whether the dataset exists or not at the given path.
        """
        path = convert_pathlib_to_string_if_needed(path)

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
        except (TokenPermissionError):
            # Cloud Dataset does not exist
            return False
        return dataset_exists(storage)

    @staticmethod
    def empty(
        path: Union[str, pathlib.Path],
        overwrite: bool = False,
        public: bool = False,
        memory_cache_size: int = DEFAULT_MEMORY_CACHE_SIZE,
        local_cache_size: int = DEFAULT_LOCAL_CACHE_SIZE,
        creds: Optional[dict] = None,
        token: Optional[str] = None,
        verbose: bool = True,
    ) -> Dataset:
        """Creates an empty dataset

        Important:
            Using `overwrite` will delete all of your data if it exists! Be very careful when setting this parameter.

        Args:
            path (str, pathlib.Path): The full path to the dataset. Can be:
                -
                - a Hub cloud path of the form `hub://username/datasetname`. To write to Hub cloud datasets, ensure that you are logged in to Hub (use 'activeloop login' from command line)
                - an s3 path of the form `s3://bucketname/path/to/dataset`. Credentials are required in either the environment or passed to the creds argument.
                - a local file system path of the form `./path/to/dataset` or `~/path/to/dataset` or `path/to/dataset`.
                - a memory path of the form `mem://path/to/dataset` which doesn't save the dataset but keeps it in memory instead. Should be used only for testing as it does not persist.
            overwrite (bool): __WARNING__: If set to True this overwrites the dataset if it already exists. This can __NOT__ be undone! Defaults to False.
            public (bool): Defines if the dataset will have public access. Applicable only if Hub cloud storage is used and a new Dataset is being created. Defaults to False.
            memory_cache_size (int): The size of the memory cache to be used in MB.
            local_cache_size (int): The size of the local filesystem cache to be used in MB.
            creds (dict, optional): - A dictionary containing credentials used to access the dataset at the path.
                - If aws_access_key_id, aws_secret_access_key, aws_session_token are present, these take precedence over credentials present in the environment or in credentials file. Currently only works with s3 paths.
                - It supports 'aws_access_key_id', 'aws_secret_access_key', 'aws_session_token', 'endpoint_url', 'aws_region', 'profile_name' as keys.
            token (str, optional): Activeloop token, used for fetching credentials to the dataset at path if it is a Hub dataset. This is optional, tokens are normally autogenerated.
            verbose (bool): If True, logs will be printed. Defaults to True.

        Returns:
            Dataset object created using the arguments provided.

        Raises:
            DatasetHandlerError: If a Dataset already exists at the given path and overwrite is False.
            UserNotLoggedInException: When user is not logged in
            InvalidTokenException: If the specified toke is invalid
            TokenPermissionError: When there are permission or other errors related to token
        """
        path = convert_pathlib_to_string_if_needed(path)

        if creds is None:
            creds = {}

        try:
            storage, cache_chain = get_storage_and_cache_chain(
                path=path,
                read_only=False,
                creds=creds,
                token=token,
                memory_cache_size=memory_cache_size,
                local_cache_size=local_cache_size,
            )

            feature_report_path(path, "empty", {"Overwrite": overwrite}, token=token)
        except Exception as e:
            if isinstance(e, UserNotLoggedInException):
                message = (
                    f"Please log in through the CLI in order to create this dataset, "
                    f"or create an API token in the UI and pass it to this method using the "
                    f"‘token’ parameter. The CLI commands are ‘activeloop login’ and ‘activeloop register’."
                )
                raise UserNotLoggedInException(message)
            raise

        if overwrite and dataset_exists(cache_chain):
            cache_chain.clear()
        elif dataset_exists(cache_chain):
            raise DatasetHandlerError(
                f"A dataset already exists at the given path ({path}). If you want to create"
                f" a new empty dataset, either specify another path or use overwrite=True. "
                f"If you want to load the dataset that exists at this path, use hub.load() instead."
            )

        read_only = storage.read_only
        return dataset_factory(
            path=path,
            storage=cache_chain,
            read_only=read_only,
            public=public,
            token=token,
            verbose=verbose,
        )

    @staticmethod
    def load(
        path: Union[str, pathlib.Path],
        read_only: Optional[bool] = None,
        memory_cache_size: int = DEFAULT_MEMORY_CACHE_SIZE,
        local_cache_size: int = DEFAULT_LOCAL_CACHE_SIZE,
        creds: Optional[dict] = None,
        token: Optional[str] = None,
        verbose: bool = True,
        access_method: str = "stream",
    ) -> Dataset:
        """Loads an existing dataset

        Args:
            path (str, pathlib.Path): - The full path to the dataset. Can be:
                - a Hub cloud path of the form `hub://username/datasetname`. To write to Hub cloud datasets, ensure that you are logged in to Hub (use 'activeloop login' from command line)
                - an s3 path of the form `s3://bucketname/path/to/dataset`. Credentials are required in either the environment or passed to the creds argument.
                - a local file system path of the form `./path/to/dataset` or `~/path/to/dataset` or `path/to/dataset`.
                - a memory path of the form `mem://path/to/dataset` which doesn't save the dataset but keeps it in memory instead. Should be used only for testing as it does not persist.
            read_only (bool, optional): Opens dataset in read only mode if this is passed as True. Defaults to False.
                Datasets stored on Hub cloud that your account does not have write access to will automatically open in read mode.
            memory_cache_size (int): The size of the memory cache to be used in MB.
            local_cache_size (int): The size of the local filesystem cache to be used in MB.
            creds (dict, optional): - A dictionary containing credentials used to access the dataset at the path.
                - If aws_access_key_id, aws_secret_access_key, aws_session_token are present, these take precedence over credentials present in the environment or in credentials file. Currently only works with s3 paths.
                - It supports 'aws_access_key_id', 'aws_secret_access_key', 'aws_session_token', 'endpoint_url', 'aws_region', 'profile_name' as keys.
            token (str, optional): Activeloop token, used for fetching credentials to the dataset at path if it is a Hub dataset. This is optional, tokens are normally autogenerated.
            verbose (bool): If True, logs will be printed. Defaults to True.
            access_method (str): - The access method to use for the dataset. Can be:-
                - 'stream' - Streams the data from the dataset i.e. only fetches data when required. This is the default.
                - 'download' -
                    - Downloads the data to the local filesystem to the path specified in environment variable HUB_DOWNLOAD_PATH.
                    - Raises an exception if the environment variable is not set, or if the path is not empty.
                    - Will also raise an exception if the dataset does not exist.
                    - The 'download' access method can also be modified to specify num_workers and/or scheduler.
                    - For example: `'download:2:processed'`, will use 2 workers and use processed scheduler, while `'download:3'` will use 3 workers and default scheduler (threaded), and `'download:processed'` will use a single worker and use processed scheduler.
                - 'local' - Used when download was already done in a previous run. Doesn't download the data again. Raises an exception if the dataset is not found in HUB_DOWNLOAD_PATH.
                Note: Any changes made to the dataset in download/local mode will only be made to the local copy and will not be reflected in the original dataset.

        Returns:
            Dataset object created using the arguments provided.

        Raises:
            DatasetHandlerError: If a Dataset does not exist at the given path.
            AgreementError: When agreement is rejected
            UserNotLoggedInException: When user is not logged in
            InvalidTokenException: If the specified toke is invalid
            TokenPermissionError: When there are permission or other errors related to token
        """
        access_method, num_workers, scheduler = parse_access_method(access_method)
        check_access_method(access_method, overwrite=False)
        path = convert_pathlib_to_string_if_needed(path)

        if creds is None:
            creds = {}

        try:
            storage, cache_chain = get_storage_and_cache_chain(
                path=path,
                read_only=read_only,
                creds=creds,
                token=token,
                memory_cache_size=memory_cache_size,
                local_cache_size=local_cache_size,
            )

            feature_report_path(path, "load", {}, token=token)
        except Exception as e:
            if isinstance(e, UserNotLoggedInException):
                message = (
                    "Please log in through the CLI in order to load this dataset, "
                    "or create an API token in the UI and pass it to this method using "
                    "the ‘token’ parameter. The CLI commands are ‘activeloop login’ and "
                    "‘activeloop register’."
                )
                raise UserNotLoggedInException(message)
            raise
        if not dataset_exists(cache_chain):
            raise DatasetHandlerError(
                f"A Hub dataset does not exist at the given path ({path}). Check the path provided or in case you want to create a new dataset, use hub.empty()."
            )

        try:
            if access_method == "stream":
                return dataset_factory(
                    path=path,
                    storage=cache_chain,
                    read_only=read_only,
                    token=token,
                    verbose=verbose,
                )
            return get_local_dataset(
                access_method=access_method,
                path=path,
                read_only=read_only,
                memory_cache_size=memory_cache_size,
                local_cache_size=local_cache_size,
                creds=creds,
                token=token,
                verbose=verbose,
                ds_exists=True,
                num_workers=num_workers,
                scheduler=scheduler,
            )
        except AgreementError as e:
            raise e from None

    @staticmethod
    def rename(
        old_path: Union[str, pathlib.Path],
        new_path: Union[str, pathlib.Path],
        creds: Optional[dict] = None,
        token: Optional[str] = None,
    ) -> Dataset:
        """Renames dataset at `old_path` to `new_path`.
        Examples:
            ```
            hub.rename("hub://username/image_ds", "hub://username/new_ds")
            hub.rename("s3://mybucket/my_ds", "s3://mybucket/renamed_ds")
            ```

        Args:
            old_path (str, pathlib.Path): The path to the dataset to be renamed.
            new_path (str, pathlib.Path): Path to the dataset after renaming.
            creds (dict, optional): - A dictionary containing credentials used to access the dataset at the path.
                - This takes precedence over credentials present in the environment. Currently only works with s3 paths.
                - It supports 'aws_access_key_id', 'aws_secret_access_key', 'aws_session_token', 'endpoint_url' and 'aws_region' as keys.
            token (str, optional): Activeloop token, used for fetching credentials to the dataset at path if it is a Hub dataset. This is optional, tokens are normally autogenerated.

        Returns:
            Dataset object after renaming.

        Raises:
            DatasetHandlerError: If a Dataset does not exist at the given path or if new path is to a different directory.
        """
        old_path = convert_pathlib_to_string_if_needed(old_path)
        new_path = convert_pathlib_to_string_if_needed(new_path)

        if creds is None:
            creds = {}

        feature_report_path(old_path, "rename", {}, token=token)

        ds = hub.load(old_path, verbose=False, token=token, creds=creds)
        ds.rename(new_path)

        return ds  # type: ignore

    @staticmethod
    def delete(
        path: Union[str, pathlib.Path],
        force: bool = False,
        large_ok: bool = False,
        creds: Optional[dict] = None,
        token: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        """Deletes a dataset at a given path.

        This is an __IRREVERSIBLE__ operation. Data once deleted can not be recovered.

        Args:
            path (str, pathlib.Path): The path to the dataset to be deleted.
            force (bool): Delete data regardless of whether
                it looks like a hub dataset. All data at the path will be removed.
            large_ok (bool): Delete datasets larger than 1GB. Disabled by default.
            creds (dict, optional): A dictionary containing credentials used to access the dataset at the path.
                -
                - If aws_access_key_id, aws_secret_access_key, aws_session_token are present, these take precedence over credentials present in the environment or in credentials file. Currently only works with s3 paths.
                - It supports 'aws_access_key_id', 'aws_secret_access_key', 'aws_session_token', 'endpoint_url', 'aws_region', 'profile_name' as keys.
            token (str, optional): Activeloop token, used for fetching credentials to the dataset at path if it is a Hub dataset. This is optional, tokens are normally autogenerated.
            verbose (bool): If True, logs will be printed. Defaults to True.

        Raises:
            DatasetHandlerError: If a Dataset does not exist at the given path and force = False.
            NotImplementedError: When attempting to delete a managed view.
        """
        path = convert_pathlib_to_string_if_needed(path)

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
            ds = hub.load(path, verbose=False, token=token, creds=creds)
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
                base_storage.clear()
                remove_path_from_backend(path, token)
                if verbose:
                    logger.info(f"{path} folder deleted successfully.")
            else:
                if isinstance(e, (DatasetHandlerError, PathNotEmptyException)):
                    raise DatasetHandlerError(
                        "A Hub dataset wasn't found at the specified path. "
                        "This may be due to a corrupt dataset or a wrong path. "
                        "If you want to delete the data at the path regardless, use force=True"
                    )
                raise

    @staticmethod
    def like(
        dest: Union[str, pathlib.Path],
        src: Union[str, Dataset, pathlib.Path],
        tensors: Optional[List[str]] = None,
        overwrite: bool = False,
        creds: Optional[dict] = None,
        token: Optional[str] = None,
        public: bool = False,
    ) -> Dataset:
        """Copies the `source` dataset's structure to a new location. No samples are copied, only the meta/info for the dataset and it's tensors.

        Args:
            dest: Empty Dataset or Path where the new dataset will be created.
            src (Union[str, Dataset]): Path or dataset object that will be used as the template for the new dataset.
            tensors (List[str], optional): Names of tensors (and groups) to be replicated. If not specified all tensors in source dataset are considered.
            overwrite (bool): If True and a dataset exists at `destination`, it will be overwritten. Defaults to False.
            creds (dict, optional): - A dictionary containing credentials used to access the dataset at the path.
                - If aws_access_key_id, aws_secret_access_key, aws_session_token are present, these take precedence over credentials present in the environment or in credentials file. Currently only works with s3 paths.
                - It supports 'aws_access_key_id', 'aws_secret_access_key', 'aws_session_token', 'endpoint_url', 'aws_region', 'profile_name' as keys.
            token (str, optional): Activeloop token, used for fetching credentials to the dataset at path if it is a Hub dataset. This is optional, tokens are normally autogenerated.
            public (bool): Defines if the dataset will have public access. Applicable only if Hub cloud storage is used and a new Dataset is being created. Defaults to False.

        Returns:
            Dataset: New dataset object.
        """
        if isinstance(dest, Dataset):
            path = dest.path
        else:
            path = dest
        feature_report_path(
            path,
            "like",
            {"Overwrite": overwrite, "Public": public, "Tensors": tensors},
            token=token,
        )
        return dataset._like(dest, src, tensors, overwrite, creds, token, public)

    @staticmethod
    def _like(  # (No reporting)
        dest,
        src: Union[str, Dataset],
        tensors: Optional[List[str]] = None,
        overwrite: bool = False,
        creds: Optional[dict] = None,
        token: Optional[str] = None,
        public: bool = False,
        unlink: Union[List[str], bool] = False,
    ) -> Dataset:
        """Copies the `source` dataset's structure to a new location. No samples are copied, only the meta/info for the dataset and it's tensors.

        Args:
            dest: Empty Dataset or Path where the new dataset will be created.
            src (Union[str, Dataset]): Path or dataset object that will be used as the template for the new dataset.
            tensors (List[str], optional): Names of tensors (and groups) to be replicated. If not specified all tensors in source dataset are considered.
            dest (str, pathlib.Path, Dataset): Empty Dataset or Path where the new dataset will be created.
            src (Union[str, pathlib.Path, Dataset]): Path or dataset object that will be used as the template for the new dataset.
            overwrite (bool): If True and a dataset exists at `destination`, it will be overwritten. Defaults to False.
            creds (dict, optional): A dictionary containing credentials used to access the dataset at the path.
                -
                - If aws_access_key_id, aws_secret_access_key, aws_session_token are present, these take precedence over credentials present in the environment or in credentials file. Currently only works with s3 paths.
                - It supports 'aws_access_key_id', 'aws_secret_access_key', 'aws_session_token', 'endpoint_url', 'aws_region', 'profile_name' as keys.
            token (str, optional): Activeloop token, used for fetching credentials to the dataset at path if it is a Hub dataset. This is optional, tokens are normally autogenerated.
            public (bool): Defines if the dataset will have public access. Applicable only if Hub cloud storage is used and a new Dataset is being created. Defaults to False.
            unlink (Union[List[str], bool]): List of tensors to be unlinked. If True passed all tensors will be unlinked. Defaults to False, no tensors are unlinked.

        Returns:
            Dataset: New dataset object.
        """
        dest = convert_pathlib_to_string_if_needed(dest)
        if isinstance(dest, Dataset):
            destination_ds = dest
            dest_path = dest.path
        else:
            dest_path = dest
            destination_ds = dataset.empty(
                dest,
                creds=creds,
                overwrite=overwrite,
                token=token,
                public=public,
            )
        feature_report_path(
            dest_path, "like", {"Overwrite": overwrite, "Public": public}, token=token
        )
        src = convert_pathlib_to_string_if_needed(src)
        if isinstance(src, str):
            source_ds = dataset.load(src)
        else:
            source_ds = src

        if tensors:
            tensors = source_ds._resolve_tensor_list(tensors)  # type: ignore
        else:
            tensors = source_ds.tensors  # type: ignore

        if unlink is True:
            unlink = tensors  # type: ignore
        elif unlink is False:
            unlink = []
        for tensor_name in tensors:  # type: ignore
            if overwrite and tensor_name in destination_ds:
                destination_ds.delete_tensor(tensor_name)
            destination_ds.create_tensor_like(tensor_name, source_ds[tensor_name], unlink=tensor_name in unlink)  # type: ignore

        destination_ds.info.update(source_ds.info.__getstate__())  # type: ignore

        return destination_ds

    @staticmethod
    def copy(
        src: Union[str, pathlib.Path, Dataset],
        dest: Union[str, pathlib.Path],
        tensors: Optional[List[str]] = None,
        overwrite: bool = False,
        src_creds=None,
        src_token=None,
        dest_creds=None,
        dest_token=None,
        num_workers: int = 0,
        scheduler="threaded",
        progressbar=True,
    ):
        """Copies this dataset at `src` to `dest`. Version control history is not included.

        Args:
            src (Union[str, Dataset, pathlib.Path]): The Dataset or the path to the dataset to be copied.
            dest (str, pathlib.Path): Destination path to copy to.
            tensors (List[str], optional): Names of tensors (and groups) to be copied. If not specified all tensors are copied.
            overwrite (bool): If True and a dataset exists at `destination`, it will be overwritten. Defaults to False.
            src_creds (dict, optional): A dictionary containing credentials used to access the dataset at `src`.
                -
                - If 'aws_access_key_id', 'aws_secret_access_key', 'aws_session_token' are present, these take precedence over credentials present in the environment or in credentials file. Currently only works with s3 paths.
                - It supports 'aws_access_key_id', 'aws_secret_access_key', 'aws_session_token', 'endpoint_url', 'aws_region', 'profile_name' as keys.
            src_token (str, optional): Activeloop token, used for fetching credentials to the dataset at `src` if it is a Hub dataset. This is optional, tokens are normally autogenerated.
            dest_creds (dict, optional): creds required to create / overwrite datasets at `dest`.
            dest_token (str, optional): token used to for fetching credentials to `dest`.
            num_workers (int): The number of workers to use for copying. Defaults to 0. When set to 0, it will always use serial processing, irrespective of the scheduler.
            scheduler (str): The scheduler to be used for copying. Supported values include: 'serial', 'threaded', 'processed' and 'ray'.
                Defaults to 'threaded'.
            progressbar (bool): Displays a progress bar if True (default).

        Returns:
            Dataset: New dataset object.

        Raises:
            DatasetHandlerError: If a dataset already exists at destination path and overwrite is False.
        """

        if isinstance(src, (str, pathlib.Path)):
            src = convert_pathlib_to_string_if_needed(src)
            src_ds = hub.load(src, read_only=True, creds=src_creds, token=src_token)
        else:
            src_ds = src
            src_ds.path = str(src_ds.path)

        dest = convert_pathlib_to_string_if_needed(dest)

        return src_ds.copy(
            dest,
            tensors=tensors,
            overwrite=overwrite,
            creds=dest_creds,
            token=dest_token,
            num_workers=num_workers,
            scheduler=scheduler,
            progressbar=progressbar,
        )

    @staticmethod
    def deepcopy(
        src: Union[str, pathlib.Path],
        dest: Union[str, pathlib.Path],
        tensors: Optional[List[str]] = None,
        overwrite: bool = False,
        src_creds=None,
        src_token=None,
        dest_creds=None,
        dest_token=None,
        num_workers: int = 0,
        scheduler="threaded",
        progressbar=True,
        public: bool = False,
        verbose: bool = True,
    ):
        """Copies dataset at `src` to `dest` including version control history.

        Args:
            src (str, pathlib.Path): Path to the dataset to be copied.
            dest (str, pathlib.Path): Destination path to copy to.
            tensors (List[str], optional): Names of tensors (and groups) to be copied. If not specified all tensors are copied.
            overwrite (bool): If True and a dataset exists at `destination`, it will be overwritten. Defaults to False.
            src_creds (dict, optional): A dictionary containing credentials used to access the dataset at `src`.
                -
                - If 'aws_access_key_id', 'aws_secret_access_key', 'aws_session_token' are present, these take precedence over credentials present in the environment or in credentials file. Currently only works with s3 paths.
                - It supports 'aws_access_key_id', 'aws_secret_access_key', 'aws_session_token', 'endpoint_url', 'aws_region', 'profile_name' as keys.
            src_token (str, optional): Activeloop token, used for fetching credentials to the dataset at `src` if it is a Hub dataset. This is optional, tokens are normally autogenerated.
            dest_creds (dict, optional): creds required to create / overwrite datasets at `dest`.
            dest_token (str, optional): token used to for fetching credentials to `dest`.
            num_workers (int): The number of workers to use for copying. Defaults to 0. When set to 0, it will always use serial processing, irrespective of the scheduler.
            scheduler (str): The scheduler to be used for copying. Supported values include: 'serial', 'threaded', 'processed' and 'ray'.
                Defaults to 'threaded'.
            progressbar (bool): Displays a progress bar if True (default).
            public (bool): Defines if the dataset will have public access. Applicable only if Hub cloud storage is used and a new Dataset is being created. Defaults to False.
            verbose (bool): If True, logs will be printed. Defaults to True.

        Returns:
            Dataset: New dataset object.

        Raises:
            DatasetHandlerError: If a dataset already exists at destination path and overwrite is False.
        """

        src = convert_pathlib_to_string_if_needed(src)
        dest = convert_pathlib_to_string_if_needed(dest)

        report_params = {
            "Overwrite": overwrite,
            "Num_Workers": num_workers,
            "Scheduler": scheduler,
            "Progressbar": progressbar,
            "Public": public,
        }
        if dest.startswith("hub://"):
            report_params["Dest"] = dest
        feature_report_path(src, "deepcopy", report_params, token=dest_token)

        src_ds = hub.load(
            src, read_only=True, creds=src_creds, token=src_token, verbose=False
        )
        src_storage = get_base_storage(src_ds.storage)

        dest_storage, cache_chain = get_storage_and_cache_chain(
            dest,
            creds=dest_creds,
            token=dest_token,
            read_only=False,
            memory_cache_size=DEFAULT_MEMORY_CACHE_SIZE,
            local_cache_size=DEFAULT_LOCAL_CACHE_SIZE,
        )

        if dataset_exists(cache_chain):
            if overwrite:
                cache_chain.clear()
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
                path=src,
            )
            for key in keys:
                val = metas.get(key) or cache[key]
                if isinstance(val, HubMemoryObject):
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
        if num_workers == 0:
            keys = [keys]
        else:
            keys = [keys[i::num_workers] for i in range(num_workers)]
        compute_provider = get_compute_provider(scheduler, num_workers)
        try:
            if progressbar:
                compute_provider.map_with_progressbar(
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
            read_only=False,
            public=public,
            token=dest_token,
            verbose=verbose,
        )
        ret._register_dataset()
        return ret

    @staticmethod
    def ingest(
        src: Union[str, pathlib.Path],
        dest: Union[str, pathlib.Path],
        sample_compression: str = "auto",
        dest_creds: dict = None,
        progressbar: bool = True,
        summary: bool = True,
        **dataset_kwargs,
    ) -> Dataset:
        """Ingests a dataset from a source and stores it as a structured dataset to destination

        Note:
            - Currently only local source paths and image classification datasets / csv files are supported for automatic ingestion.
            - Supported filetypes: png/jpeg/jpg/csv.
            - All files and sub-directories with unsupported filetypes are ignored.
            - Valid source directory structures for image classification look like:

            ```
                data/
                    img0.jpg
                    img1.jpg
                    ...

            ```
            or
            ```
                data/
                    class0/
                        cat0.jpg
                        ...
                    class1/
                        dog0.jpg
                        ...
                    ...

            ```
            or
            ```
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
            ```

            - Classes defined as sub-directories can be accessed at `ds["test/labels"].info.class_names`.
            - Support for train and test sub directories is present under ds["train/images"], ds["train/labels"] and ds["test/images"], ds["test/labels"]
            - Mapping filenames to classes from an external file is currently not supported.

        Args:
            src (str, pathlib.Path): Local path to where the unstructured dataset is stored or path to csv file.
            dest (str, pathlib.Path): - Destination path where the structured dataset will be stored. Can be:
                - a Hub cloud path of the form `hub://username/datasetname`. To write to Hub cloud datasets, ensure that you are logged in to Hub (use 'activeloop login' from command line)
                - an s3 path of the form `s3://bucketname/path/to/dataset`. Credentials are required in either the environment or passed to the creds argument.
                - a local file system path of the form `./path/to/dataset` or `~/path/to/dataset` or `path/to/dataset`.
                - a memory path of the form `mem://path/to/dataset` which doesn't save the dataset but keeps it in memory instead. Should be used only for testing as it does not persist.
            sample_compression (str): For image classification datasets, this compression will be used for the `images` tensor. If sample_compression is "auto", compression will be automatically determined by the most common extension in the directory.
            dest_creds (dict): A dictionary containing credentials used to access the destination path of the dataset.
            progressbar (bool): Enables or disables ingestion progress bar. Defaults to True.
            summary (bool): If True, a summary of skipped files will be printed after completion. Defaults to True.
            **dataset_kwargs: Any arguments passed here will be forwarded to the dataset creator function.

        Returns:
            Dataset: New dataset object with structured dataset.

        Raises:
            InvalidPathException: If the source directory does not exist.
            SamePathException: If the source and destination path are same.
            AutoCompressionError: If the source director is empty or does not contain a valid extension.
            InvalidFileExtension: If the most frequent file extension is found to be 'None' during auto-compression.
        """
        dest = convert_pathlib_to_string_if_needed(dest)
        feature_report_path(
            dest,
            "ingest",
            {
                "sample_compression": sample_compression,
                "Progressbar": progressbar,
                "Summary": summary,
            },
        )

        src = convert_pathlib_to_string_if_needed(src)

        if isinstance(src, str):
            if os.path.isdir(dest) and os.path.samefile(src, dest):
                raise SamePathException(src)

            if src.endswith(".csv"):
                import pandas as pd  # type:ignore

                if not os.path.isfile(src):
                    raise InvalidPathException(src)
                source = pd.read_csv(src, quotechar='"', skipinitialspace=True)
                ds = dataset.ingest_dataframe(
                    source, dest, dest_creds, progressbar, **dataset_kwargs
                )
                return ds

            if not os.path.isdir(src):
                raise InvalidPathException(src)

            if sample_compression == "auto":
                sample_compression = get_most_common_extension(src)
                if sample_compression is None:
                    raise InvalidFileExtension(src)
            ds = hub.dataset(dest, creds=dest_creds, **dataset_kwargs)

            # TODO: support more than just image classification (and update docstring)
            if sample_compression in _image_compressions:
                unstructured = ImageClassification(source=src, htype="image")  # type: ignore
            elif sample_compression in _audio_compressions:
                unstructured = AudioClassification(source=src, htype="audio")  # type: ignore
            elif sample_compression in _video_compressions:
                unstructured = VideoClassification(source=src, htype="video")  # type: ignore

            # TODO: auto detect compression
            unstructured.structure(
                ds,  # type: ignore
                use_progress_bar=progressbar,
                generate_summary=summary,
                tensor_args={"sample_compression": sample_compression},
            )
        return ds  # type: ignore

    @staticmethod
    def ingest_kaggle(
        tag: str,
        src: Union[str, pathlib.Path],
        dest: Union[str, pathlib.Path],
        exist_ok: bool = False,
        sample_compression: str = "auto",
        dest_creds: dict = None,
        kaggle_credentials: dict = None,
        progressbar: bool = True,
        summary: bool = True,
        **dataset_kwargs,
    ) -> Dataset:
        """Download and ingest a kaggle dataset and store it as a structured dataset to destination

        Note:
            Currently only local source paths and image classification datasets are supported for automatic ingestion.

        Args:
            tag (str): Kaggle dataset tag. Example: `"coloradokb/dandelionimages"` points to https://www.kaggle.com/coloradokb/dandelionimages
            src (str, pathlib.Path): Local path to where the raw kaggle dataset will be downlaoded to.
            dest (str, pathlib.Path): - Destination path where the structured dataset will be stored. Can be:
                - a Hub cloud path of the form `hub://username/datasetname`. To write to Hub cloud datasets, ensure that you are logged in to Hub (use 'activeloop login' from command line)
                - an s3 path of the form `s3://bucketname/path/to/dataset`. Credentials are required in either the environment or passed to the creds argument.
                - a local file system path of the form `./path/to/dataset` or `~/path/to/dataset` or `path/to/dataset`.
                - a memory path of the form `mem://path/to/dataset` which doesn't save the dataset but keeps it in memory instead. Should be used only for testing as it does not persist.
            exist_ok (bool): If the kaggle dataset was already downloaded and `exist_ok` is True, ingestion will proceed without error.
            sample_compression (str): For image classification datasets, this compression will be used for the `images` tensor. If sample_compression is "auto", compression will be automatically determined by the most common extension in the directory.
            dest_creds (dict): A dictionary containing credentials used to access the destination path of the dataset.
            kaggle_credentials (dict): A dictionary containing kaggle credentials {"username":"YOUR_USERNAME", "key": "YOUR_KEY"}. If None, environment variables/the kaggle.json file will be used if available.
            progressbar (bool): Enables or disables ingestion progress bar. Set to true by default.
            summary (bool): Generates ingestion summary. Set to true by default.
            **dataset_kwargs: Any arguments passed here will be forwarded to the dataset creator function.

        Returns:
            Dataset: New dataset object with structured dataset.

        Raises:
            SamePathException: If the source and destination path are same.
        """
        src = convert_pathlib_to_string_if_needed(src)
        dest = convert_pathlib_to_string_if_needed(dest)

        feature_report_path(
            dest,
            "ingest_kaggle",
            {
                "sample_compression": sample_compression,
                "Exist_Ok": exist_ok,
                "Progressbar": progressbar,
                "Summary": summary,
            },
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

        ds = hub.ingest(
            src=src,
            dest=dest,
            sample_compression=sample_compression,
            dest_creds=dest_creds,
            progressbar=progressbar,
            summary=summary,
            **dataset_kwargs,
        )

        return ds

    @staticmethod
    def ingest_dataframe(
        src,
        dest: Union[str, pathlib.Path],
        dest_creds: dict = None,
        progressbar: bool = True,
        **dataset_kwargs,
    ):
        import pandas as pd
        from hub.auto.structured.dataframe import DataFrame

        if not isinstance(src, pd.DataFrame):
            raise Exception("Source provided is not a valid pandas dataframe object")

        dest = convert_pathlib_to_string_if_needed(dest)

        ds = hub.dataset(dest, creds=dest_creds, **dataset_kwargs)

        structured = DataFrame(src)
        structured.fill_dataset(ds, progressbar)  # type: ignore
        return ds  # type: ignore

    @staticmethod
    @hub_reporter.record_call
    def list(
        workspace: str = "",
        token: Optional[str] = None,
    ) -> None:
        """List all available hub cloud datasets.

        Args:
            workspace (str): Specify user/organization name. If not given,
                returns a list of all datasets that can be accessed, regardless of what workspace they are in.
                Otherwise, lists all datasets in the given workspace.
            token (str, optional): Activeloop token, used for fetching credentials for Hub datasets. This is optional, tokens are normally autogenerated.

        Returns:
            List of dataset names.
        """
        client = HubBackendClient(token=token)
        datasets = client.get_datasets(workspace=workspace)
        return datasets
