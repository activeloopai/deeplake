from typing import Callable, Dict, Optional, Union, Tuple, List
from hub.api.tensor import Tensor
from hub.constants import (
    DEFAULT_MEMORY_CACHE_SIZE,
    DEFAULT_LOCAL_CACHE_SIZE,
    MB,
)
from hub.core.dataset import dataset_exists
from hub.core.meta.dataset_meta import (
    read_dataset_meta,
    write_dataset_meta,
    default_dataset_meta,
)
from hub.core.meta.tensor_meta import default_tensor_meta
from hub.core.tensor import tensor_exists
from hub.core.typing import StorageProvider
from hub.core.index import Index
from hub.integrations import dataset_to_pytorch, dataset_to_tensorflow
from hub.util.cache_chain import generate_chain
from hub.util.exceptions import (
    InvalidKeyTypeError,
    PathNotEmptyException,
    TensorAlreadyExistsError,
    TensorDoesNotExistError,
)
from hub.util.get_storage_provider import get_storage_provider
from hub.util.path import get_path_from_storage


class Dataset:
    def __init__(
        self,
        path: Optional[str] = None,
        read_only: bool = False,
        index: Index = Index(),
        memory_cache_size: int = DEFAULT_MEMORY_CACHE_SIZE,
        local_cache_size: int = DEFAULT_LOCAL_CACHE_SIZE,
        creds: Optional[dict] = None,
        storage: Optional[StorageProvider] = None,
    ):
        """Initializes a new or existing dataset.

        Args:
            path (str, optional): The full path to the dataset.
                Can be a Hub cloud path of the form hub://username/datasetname. To write to Hub cloud datasets, ensure that you are logged in to Hub (use 'activeloop login' from command line)
                Can be a s3 path of the form s3://bucketname/path/to/dataset. Credentials are required in either the environment or passed to the creds argument.
                Can be a local file system path of the form ./path/to/dataset or ~/path/to/dataset or path/to/dataset.
                Can be a memory path of the form mem://path/to/dataset which doesn't save the dataset but keeps it in memory instead. Should be used only for testing as it does not persist.
            read_only (bool): Opens dataset in read only mode if this is passed as True. Defaults to False.
                Datasets stored on Hub cloud that your account does not have write access to will automatically open in read mode.
            index (Index): The Index object restricting the view of this dataset's tensors.
            memory_cache_size (int): The size of the memory cache to be used in MB.
            local_cache_size (int): The size of the local filesystem cache to be used in MB.
            creds (dict, optional): A dictionary containing credentials used to access the dataset at the path.
                This takes precedence over credentials present in the environment. Currently only works with s3 paths.
                It supports 'aws_access_key_id', 'aws_secret_access_key', 'aws_session_token', 'endpoint_url' and 'region' as keys.
            storage (StorageProvider, optional): The storage provider used to access the dataset.
                Use this if you want to specify the storage provider object manually instead of using a path to generate it.

        Raises:
            ValueError: If an existing local path is given, it must be a directory.
            ImproperDatasetInitialization: Exactly one argument out of 'path' and 'storage' needs to be specified.
                This is raised if none of them are specified or more than one are specifed.
            InvalidHubPathException: If a Hub cloud path (path starting with hub://) is specified and it isn't of the form hub://username/datasetname.
            AuthorizationException: If a Hub cloud path (path starting with hub://) is specified and the user doesn't have access to the dataset.
            PathNotEmptyException: If the path to the dataset doesn't contain a Hub dataset and is also not empty.
        """
        if creds is None:
            creds = {}
        base_storage = get_storage_provider(path, storage, read_only, creds)

        # done instead of directly assigning read_only as backend might return return read_only permissions
        if hasattr(base_storage, "read_only") and base_storage.read_only:
            self.read_only = True
        else:
            self.read_only = False

        # uniquely identifies dataset
        self.path = path or get_path_from_storage(base_storage)
        memory_cache_size_bytes = memory_cache_size * MB
        local_cache_size_bytes = local_cache_size * MB
        self.storage = generate_chain(
            base_storage, memory_cache_size_bytes, local_cache_size_bytes, path
        )
        self.storage.autoflush = True
        self.index = index

        self.tensors: Dict[str, Tensor] = {}
        if dataset_exists(self.storage):
            for tensor_name in self.meta["tensors"]:
                self.tensors[tensor_name] = Tensor(tensor_name, self.storage)
        elif len(self.storage) > 0:
            raise PathNotEmptyException
        else:
            self.meta = default_dataset_meta()

    def __enter__(self):
        self.storage.autoflush = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.storage.autoflush = True
        self.flush()

    # TODO len should consider slice
    def __len__(self):
        """Return the smallest length of tensors"""
        return min(map(len, self.tensors.values()), default=0)

    def __getitem__(
        self,
        item: Union[
            str, int, slice, List[int], Tuple[Union[int, slice, Tuple[int]]], Index
        ],
    ):
        if isinstance(item, str):
            if item not in self.tensors:
                raise TensorDoesNotExistError(item)
            else:
                return self.tensors[item][self.index]
        elif isinstance(item, (int, slice, list, tuple, Index)):
            return Dataset(
                read_only=self.read_only,
                storage=self.storage,
                index=self.index[item],
            )
        else:
            raise InvalidKeyTypeError(item)

    def create_tensor(
        self,
        name: str,
        htype: Optional[str] = None,
        chunk_size: Optional[int] = None,
        dtype: Optional[str] = None,
        extra_meta: Optional[dict] = None,
    ):
        """Creates a new tensor in a dataset.

        Args:
            name (str): The name of the tensor to be created.
            htype (str, optional): The class of data for the tensor.
                The defaults for other parameters are determined in terms of this value.
                For example, `htype="image"` would have `dtype` default to `uint8`.
                These defaults can be overridden by explicitly passing any of the other parameters to this function.
                May also modify the defaults for other parameters.
            chunk_size (int, optional): The target size for chunks in this tensor.
            dtype (str, optional): The data type to use for this tensor.
                Will be overwritten when the first sample is added.
            extra_meta (dict, optional): Any additional metadata to be added to the tensor.

        Returns:
            The new tensor, which can also be accessed by `self[name]`.

        Raises:
            TensorAlreadyExistsError: Duplicate tensors are not allowed.
        """
        if tensor_exists(name, self.storage):
            raise TensorAlreadyExistsError(name)

        ds_meta = self.meta
        ds_meta["tensors"].append(name)
        self.meta = ds_meta

        tensor_meta = default_tensor_meta(htype, chunk_size, dtype, extra_meta)
        tensor = Tensor(name, self.storage, tensor_meta=tensor_meta)
        self.tensors[name] = tensor

        return tensor

    __getattr__ = __getitem__

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @property
    def meta(self):
        return read_dataset_meta(self.storage)

    @meta.setter
    def meta(self, new_meta: dict):
        write_dataset_meta(self.storage, new_meta)

    def pytorch(self, transform: Optional[Callable] = None, workers: int = 1):
        """Converts the dataset into a pytorch compatible format.

        Note:
            Pytorch does not support uint16, uint32, uint64 dtypes. These are implicitly type casted to int32, int64 and int64 respectively.
            This spins up it's own workers to fetch data, when using with torch.utils.data.DataLoader, set num_workers = 0 to avoid issues.

        Args:
            transform (Callable, optional) : Transformation function to be applied to each sample
            workers (int): The number of workers to use for fetching data in parallel.

        Returns:
            A dataset object that can be passed to torch.utils.data.DataLoader
        """
        return dataset_to_pytorch(self, transform, workers=workers)

    def tensorflow(self):
        """Converts the dataset into a pytorch compatible format.

        Returns:
            tf.data.Dataset object that can be used for tensorflow training.
        """
        return dataset_to_tensorflow(self)

    def flush(self):
        """Necessary operation after writes if caches are being used.
        Writes all the dirty data from the cache layers (if any) to the underlying storage.
        Here dirty data corresponds to data that has been changed/assigned and but hasn't yet been sent to the
        underlying storage.
        """
        self.storage.flush()

    def clear_cache(self):
        """Flushes (see Dataset.flush documentation) the contents of the cache layers (if any) and then deletes contents
         of all the layers of it.
        This doesn't delete data from the actual storage.
        This is useful if you have multiple datasets with memory caches open, taking up too much RAM.
        Also useful when local cache is no longer needed for certain datasets and is taking up storage space.
        """
        if hasattr(self.storage, "clear_cache"):
            self.storage.clear_cache()

    def delete(self):
        """Deletes the entire dataset from the cache layers (if any) and the underlying storage.
        This is an IRREVERSIBLE operation. Data once deleted can not be recovered.
        """
        self.storage.clear()

    @staticmethod
    def from_path(path: str):
        """Creates a hub dataset from unstructured data.

        Note:
            This copies the data into hub format.
            Be careful when using this with large datasets.

        Args:
            path (str): Path to the data to be converted

        Returns:
            A Dataset instance whose path points to the hub formatted
            copy of the data.

        Raises:
            NotImplementedError: TODO.
        """

        raise NotImplementedError(
            "Automatic dataset ingestion is not yet supported."
        )  # TODO: hub.auto
        return None

    def __str__(self):
        path_str = ""
        if self.path:
            path_str = f"path={self.path}, "

        mode_str = ""
        if self.read_only:
            mode_str = f"read_only=True, "

        index_str = f"index={self.index}, "
        if self.index.is_trivial():
            index_str = ""

        return f"Dataset({path_str}{mode_str}{index_str}tensors={self.meta['tensors']})"
