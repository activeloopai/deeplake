from hub.util.cache_chain import generate_chain
from hub.constants import (
    MB,
    META_FILENAME,
    DEFAULT_MEMORY_CACHE_SIZE,
    DEFAULT_LOCAL_CACHE_SIZE,
)
from hub.api.tensor import Tensor
from hub.util.slice import merge_slices
from hub.util.path import storage_provider_from_path
from hub.util.exceptions import (
    TensorNotFoundError,
    InvalidKeyTypeError,
    UnsupportedTensorTypeError,
)
from hub.core.typing import StorageProvider
from hub.core.chunk_engine.read import read_dataset_meta, read_tensor_meta
from hub.core.chunk_engine.write import write_array, write_dataset_meta
from typing import Union, Dict, Optional
import numpy as np
import warnings


class Dataset:
    def __init__(
        self,
        path: str = "",
        mode: str = "a",
        ds_slice: slice = slice(None),
        memory_cache_size: int = DEFAULT_MEMORY_CACHE_SIZE,
        local_cache_size: int = DEFAULT_LOCAL_CACHE_SIZE,
        storage: Optional[StorageProvider] = None,
    ):
        """Initialize a new or existing dataset.

        Args:
            path (str): The location of the dataset. Used to initialize the storage provider.
            mode (str): Mode in which the dataset is opened.
                Supported modes include ("r", "w", "a") plus an optional "+" suffix.
                Defaults to "a".
            ds_slice (slice): The slice object restricting the view of this dataset's tensors.
                Defaults to slice(None, None, None). Used internally for iteration.
            memory_cache_size (int): The size of the memory cache to be used in MB.
            local_cache_size (int): The size of the local filesystem cache to be used in MB.
            storage (StorageProvider, optional): The storage provider used to access
            the data stored by this dataset. If this is specified, the path given is ignored.


        Raises:
            ValueError: If an existing local path is given, it must be a directory.
            UserWarning: Both path and storage should not be given.
        """
        self.mode = mode
        self.slice = ds_slice

        if storage is not None and path:
            warnings.warn(
                "Dataset should not be constructed with both storage and path. Ignoring path and using storage."
            )
        base_storage = storage or storage_provider_from_path(path)
        memory_cache_size_bytes = memory_cache_size * MB
        local_cache_size_bytes = local_cache_size * MB
        self.storage = generate_chain(
            base_storage, memory_cache_size_bytes, local_cache_size_bytes, path
        )
        self.tensors: Dict[str, Tensor] = {}
        if META_FILENAME in self.storage:
            ds_meta = read_dataset_meta(self.storage)
            for tensor_name in ds_meta["tensors"]:
                self.tensors[tensor_name] = Tensor(tensor_name, self.storage)
        else:
            write_dataset_meta(self.storage, {"tensors": []})

    def __len__(self):
        """Return the greatest length of tensors"""
        return max(map(len, self.tensors.values()), default=0)

    def __getitem__(self, item: Union[slice, str, int]):
        if isinstance(item, int):
            item = slice(item, item + 1)

        if isinstance(item, str):
            if item not in self.tensors:
                raise TensorNotFoundError(item)
            else:
                return self.tensors[item][self.slice]
        elif isinstance(item, slice):
            new_slice = merge_slices(self.slice, item)
            return Dataset(mode=self.mode, ds_slice=new_slice, storage=self.storage)
        else:
            raise InvalidKeyTypeError(item)

    def __setitem__(self, item: Union[slice, str], value):
        if isinstance(item, str):
            if isinstance(value, np.ndarray):
                write_array(
                    value,
                    item,
                    storage=self.storage,
                    batched=True,
                )
                ds_meta = read_dataset_meta(self.storage)
                ds_meta["tensors"].append(item)
                write_dataset_meta(self.storage, ds_meta)
                self.tensors[item] = Tensor(item, self.storage)
                return self.tensors[item]
            else:
                raise UnsupportedTensorTypeError(item)
        else:
            raise InvalidKeyTypeError(item)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def flush(self):
        """Necessary operation after writes if caches are being used. 
        Writes all the dirty data from the cache layers (if any) to the underlying storage.
        Here dirty data corresponds to data that has been changed/assigned and but hasn't yet been sent to the underlying storage.
        """
        self.storage.flush()

    def clear_cache(self):
        """Flushes (see Dataset.flush documentation) the contents of the cache layers (if any) and then deletes contents of all the layers of it.
        This doesn't delete data from the actual storage.
        This is useful if you have multiple datasets with memory caches open, taking up too much RAM.
        Also useful when local cache is no longer needed for certain datasets and is taking up storage space.
        """
        if self.storage.hasattr("clear_cache"):
            self.storage.clear_cache()

    def delete(self):
        """Deletes the entire dataset from the cache layers (if any) and the underlying storage. 
        This is an IRREVERSIBLE operation. Data once deleted can not be recovered.
        """
        self.storage.clear()

    @staticmethod
    def from_path(path: str):
        """Create a local hub dataset from unstructured data.

        Note:
            This copies the data locally in hub format.
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
