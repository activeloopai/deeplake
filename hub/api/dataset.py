import shutil
from hub.api.unstructured_dataset.image_classification import ImageClassification
from hub.core.storage.memory import MemoryProvider
from hub.htypes import DEFAULT_HTYPE
import warnings
from typing import Callable, Dict, Optional, Union, Tuple, List

from hub.api.tensor import Tensor
from hub.constants import DEFAULT_MEMORY_CACHE_SIZE, DEFAULT_LOCAL_CACHE_SIZE, MB
from hub.core.dataset import dataset_exists
from hub.core.meta.dataset_meta import DatasetMeta
from hub.core.tensor import create_tensor, tensor_exists
from hub.core.typing import StorageProvider
from hub.core.index import Index
from hub.integrations import dataset_to_pytorch
from hub.util.cache_chain import generate_chain
from hub.util.kaggle import download_kaggle_dataset
from hub.util.exceptions import (
    InvalidKeyTypeError,
    KaggleDatasetAlreadyDownloadedError,
    TensorAlreadyExistsError,
    TensorDoesNotExistError,
)
from hub.util.path import storage_provider_from_path


def _get_cache_chain(
    path: str,
    storage: StorageProvider,
    memory_cache_size: int,
    local_cache_size: int,
    **kwargs,
):
    if storage is not None and path:
        warnings.warn(
            "Dataset should not be constructed with both storage and path. Ignoring path and using storage."
        )

    if isinstance(storage, MemoryProvider):
        return storage

    base_storage = storage or storage_provider_from_path(path)
    memory_cache_size_bytes = memory_cache_size * MB
    local_cache_size_bytes = local_cache_size * MB
    return generate_chain(
        base_storage, memory_cache_size_bytes, local_cache_size_bytes, path
    )


class Dataset:
    def __init__(
        self,
        path: str = "",
        mode: str = "a",
        index: Index = Index(),
        memory_cache_size: int = DEFAULT_MEMORY_CACHE_SIZE,
        local_cache_size: int = DEFAULT_LOCAL_CACHE_SIZE,
        storage: Optional[StorageProvider] = None,
    ):
        """Initializes a new or existing dataset.

        Args:
            path (str): The location of the dataset. Used to initialize the storage provider.
            mode (str): Mode in which the dataset is opened.
                Supported modes include ("r", "w", "a").
                Defaults to "a".
            index (Index): The Index object restricting the view of this dataset's tensors.
            memory_cache_size (int): The size of the memory cache to be used in MB.
            local_cache_size (int): The size of the local filesystem cache to be used in MB.
            storage (StorageProvider, optional): The storage provider used to access
                the data stored by this dataset. If this is specified, the path given is ignored.

        Raises:
            ValueError: If an existing local path is given, it must be a directory.
            UserWarning: Both path and storage should not be given.
        """
        self.mode = mode
        self.index = index
        self.path = path  # Used for printing, if given

        if storage is not None and hasattr(storage, "root"):
            # Extract the path for printing, if path not given
            self.path = storage.root  # type: ignore

        self.storage = _get_cache_chain(
            path, storage, memory_cache_size, local_cache_size
        )
        self.tensors: Dict[str, Tensor] = {}

        self.tensors: Dict[str, Tensor] = {}
        if dataset_exists(self.storage):
            self.meta = DatasetMeta.load(self.storage)
            for tensor_name in self.meta.tensors:
                self.tensors[tensor_name] = Tensor(tensor_name, self.storage)
        else:
            self.meta = DatasetMeta.create(self.storage)

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
            return Dataset(mode=self.mode, storage=self.storage, index=self.index[item])
        else:
            raise InvalidKeyTypeError(item)

    def create_tensor(
        self,
        name: str,
        htype: str = DEFAULT_HTYPE,
        **kwargs,
    ):
        """Creates a new tensor in a dataset.

        Args:
            name (str): The name of the tensor to be created.
            htype (str): The class of data for the tensor.
                The defaults for other parameters are determined in terms of this value.
                For example, `htype="image"` would have `dtype` default to `uint8`.
                These defaults can be overridden by explicitly passing any of the other parameters to this function.
                May also modify the defaults for other parameters.
            **kwargs: `htype` defaults can be overridden by passing any of the compatible parameters.
                To see all `htype`s and their correspondent arguments, check out `hub/htypes.py`.

        Returns:
            The new tensor, which can also be accessed by `self[name]`.

        Raises:
            TensorAlreadyExistsError: Duplicate tensors are not allowed.
        """

        if tensor_exists(name, self.storage):
            raise TensorAlreadyExistsError(name)

        create_tensor(name, self.storage, htype=htype, **kwargs)
        tensor = Tensor(name, self.storage)

        self.tensors[name] = tensor
        self.meta.tensors.append(name)

        return tensor

    __getattr__ = __getitem__

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

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

    def keys(self):
        return tuple(self.tensors.keys())

    @staticmethod
    def from_path(
        source: str,
        destination: Union[str, StorageProvider],
        delete_source: bool = False,
        use_progress_bar: bool = True,
        **kwargs,
    ):
        """Copies unstructured data from `source` and structures/sends it to `destination`.

        Note:
            Be careful when providing sources to large datasets!
            This method copies data from `source` to `destination`.
            To be safe, you should assume the size of your dataset will consume 3-5x more space than expected.

        Args:
            source (str): Local-only path to where the unstructured dataset is stored.
            destination (str | StorageProvider): Path/StorageProvider where the structured data will be stored.
            delete_source (bool): WARNING: effectively calling `rm -rf {source}`. Deletes the entire contents of `source`
                after ingestion is complete.
            use_progress_bar (bool): If True, a progress bar is used for ingestion progress.
            **kwargs: Args will be passed into `hub.Dataset`.

        Returns:
            A read-only `hub.Dataset` instance pointing to the structured data.
        """

        # TODO: make sure source and destination paths are not equal

        _warn_kwargs("from_path", **kwargs)

        if isinstance(destination, StorageProvider):
            kwargs["storage"] = destination
        else:
            kwargs["path"] = destination

        # TODO: check for incomplete ingestion
        # TODO: try to resume progress for incomplete ingestion

        # TODO: this is not properly working (write a test for this too). expected it to pick up already structured datasets, but it doesn't
        if _dataset_has_tensors(**kwargs):
            return Dataset(**kwargs, mode="r")

        ds = Dataset(**kwargs, mode="w")

        # TODO: auto detect which `UnstructuredDataset` subclass to use
        unstructured = ImageClassification(source)
        unstructured.structure(ds, use_progress_bar=use_progress_bar)

        if delete_source:
            shutil.rmtree(source)

        return Dataset(**kwargs, mode="r")

    @staticmethod
    def from_kaggle(
        tag: str,
        source: str,
        destination: Union[str, StorageProvider],
        kaggle_credentials: dict = {},
        **kwargs,
    ):

        """Downloads the kaggle dataset with the given `tag` to this local machine, then that data is structured and copied into `destination`.

        Note:
            Be careful when providing tags to large datasets!
            This method downloads data from kaggle to the calling machine's local storage.
            To be safe, you should assume the size of the kaggle dataset being downloaded will consume 3-5x more space than expected.

        Args:
            tag (str): Kaggle dataset tag. Example: `"coloradokb/dandelionimages"` points to https://www.kaggle.com/coloradokb/dandelionimages
            source (str): Local-only path to where the unstructured kaggle dataset will be downloaded/unzipped.
            destination (str | StorageProvider): Path/StorageProvider where the structured data will be stored.
            kaggle_credentials (dict): Kaggle credentials, can directly copy and paste directly from the `kaggle.json` that is generated by kaggle.
                For more information, check out https://www.kaggle.com/docs/api
                Expected dict keys: ["username", "key"].
            **kwargs: Args will be passed into `from_path`.

        Returns:
            A read-only `hub.Dataset` instance pointing to the structured data.
        """

        _warn_kwargs("from_kaggle", **kwargs)

        if _dataset_has_tensors(**kwargs):
            return Dataset(**kwargs, mode="r")

        try:
            download_kaggle_dataset(
                tag, local_path=source, kaggle_credentials=kaggle_credentials
            )
        except KaggleDatasetAlreadyDownloadedError as e:
            warnings.warn(e.message)

        ds = Dataset.from_path(source=source, destination=destination, **kwargs)

        return ds

    def __str__(self):
        path_str = f"path={self.path}, "
        if not self.path:
            path_str = ""
        index_str = f"index={self.index}, "
        if self.index.is_trivial():
            index_str = ""
        return f"Dataset({path_str}mode={repr(self.mode)}, {index_str}tensors={self.meta.tensors})"


def _dataset_has_tensors(**kwargs):
    ds = Dataset(**kwargs, mode="r")
    return len(ds.keys()) > 0


def _warn_kwargs(caller: str, **kwargs):
    if _dataset_has_tensors(**kwargs):
        warnings.warn(
            "Dataset already exists, skipping ingestion and returning a read-only Dataset."
        )
        return  # no other warnings should print

    if "mode" in kwargs:
        warnings.warn(
            'Argument `mode` should not be passed to `%s`. Ignoring and using `mode="write"`.'
            % caller
        )

    if "path" in kwargs:
        # TODO: generalize warns
        warnings.warn(
            "Argument `path` should not be passed to `%s`. Ignoring and using `destination`."
            % caller
        )
