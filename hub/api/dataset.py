import warnings
from typing import Callable, Dict, Optional, Union, Tuple, List

from hub.api.tensor import Tensor
from hub.constants import DEFAULT_MEMORY_CACHE_SIZE, DEFAULT_LOCAL_CACHE_SIZE, MB
from hub.core.dataset import dataset_exists
from hub.core.meta.dataset_meta import read_dataset_meta, write_dataset_meta
from hub.core.meta.tensor_meta import default_tensor_meta
from hub.core.tensor import tensor_exists
from hub.core.typing import StorageProvider
from hub.core.index import Index
from hub.constants import DEFAULT_CHUNK_SIZE
from hub.integrations import dataset_to_pytorch
from hub.util.cache_chain import generate_chain
from hub.util.exceptions import (
    InvalidKeyTypeError,
    PathNotEmptyException,
    TensorAlreadyExistsError,
    TensorDoesNotExistError,
)
from hub.util.index import Index
from hub.util.get_storage_provider import get_storage_provider


class Dataset:
    def __init__(
        self,
        tag: Optional[str] = None,
        url: Optional[str] = None,
        mode: Optional[str] = None,
        index: Index = Index(),
        memory_cache_size: int = DEFAULT_MEMORY_CACHE_SIZE,
        local_cache_size: int = DEFAULT_LOCAL_CACHE_SIZE,
        creds: Optional[dict] = None,
        storage: Optional[StorageProvider] = None,
    ):
        """Initializes a new or existing dataset.

        Args:
            tag (str, optional): The Hub tag of the dataset in the format username/datasetname. Use this if you want to use Hub cloud storage.
            url (str, optional): The full path to the dataset. Use this if you want to use local filesystem or s3 or RAM to store the dataset.
                Can be either a s3 path of the form s3://bucketname/path/to/dataset. Credentials are required in either the environment or passed to the creds argument.
                Can be a local file system path of the form ./path/to/dataset or ~/path/to/dataset or path/to/dataset.
                Can be a memory path of the form mem://path/to/dataset which doesn't save the dataset but keeps it in memory instead. Should be used only for testing as it does not persist.
            mode (str, optional): Mode in which the dataset is opened.
                Supported modes include ("r", "w", "a").
            index (Index): The Index object restricting the view of this dataset's tensors.
            memory_cache_size (int): The size of the memory cache to be used in MB.
            local_cache_size (int): The size of the local filesystem cache to be used in MB.
            creds (dict, optional): A dictionary containing credentials used to access the dataset at the url.
                This takes precedence over credentials present in the environment. Only used when url is provided. Currently only works with s3 urls.
                It supports 'aws_access_key_id', 'aws_secret_access_key', 'aws_session_token', 'endpoint_url' and 'region' as keys.
            storage (StorageProvider, optional): The storage provider used to access the dataset.
                Use this if you want to specify the storage provider object manually instead of using a tag or url to generate it.

        Raises:
            ValueError: If an existing local path is given, it must be a directory.
            ImproperDatasetInitialization: Exactly one argument out of 'tag', 'url' and 'storage' needs to be specified.
                This is raised if none of them are specified or more than one are specifed.
            InvalidTagException: If an incorrect tag argument is passed which is not in username/datasetname format.
            AuthorizationException: If a tag is specified and the user doesn't have access to the dataset.
        """
        self.index = index
        if creds is None:
            creds = {}
        base_storage = get_storage_provider(tag, url, storage, mode, creds)
        memory_cache_size_bytes = memory_cache_size * MB
        local_cache_size_bytes = local_cache_size * MB
        self.storage = generate_chain(
            base_storage, memory_cache_size_bytes, local_cache_size_bytes, url
        )
        self.tensors: Dict[str, Tensor] = {}

        if dataset_exists(self.storage):
            for tensor_name in self.meta["tensors"]:
                self.tensors[tensor_name] = Tensor(tensor_name, self.storage)
        elif len(self.storage) > 0:
            raise PathNotEmptyException
        else:
            self.meta = {"tensors": []}

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
            return Dataset(storage=self.storage, index=self.index[item])
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
