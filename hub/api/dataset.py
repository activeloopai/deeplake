from hub.constants import META_FILENAME
from hub.api.tensor import Tensor
from hub.util.index import Index
from hub.util.path import provider_from_path
from hub.util.exceptions import (
    TensorNotFoundError,
    InvalidKeyTypeError,
    UnsupportedTensorTypeError,
)
from hub.core.typing import StorageProvider
from hub.core.storage import MemoryProvider
from hub.core.chunk_engine.read import read_dataset_meta, read_tensor_meta
from hub.core.chunk_engine.write import write_array, write_dataset_meta
from typing import Union, Dict, Optional
import numpy as np
import warnings
import os

# Used to distinguish between attributes and items (tensors)
DATASET_RESERVED_ATTRIBUTES = ["path", "mode", "index", "provider", "tensors"]


class Dataset:
    def __init__(
        self,
        path: str = "",
        mode: str = "a",
        provider: Optional[StorageProvider] = None,
        index: Union[int, slice, Index] = None,
    ):
        """Initialize a new or existing dataset.

        Note:
            Entries of `DATASET_RESERVED_ATTRIBUTES` cannot be used as tensor names.
            This is to distinguish between attributes (like `ds.mode`) and tensors.

            Be sure to keep `DATASET_RESERVED_ATTRIBUTES` up-to-date when changing this class.

        Args:
            path (str): The location of the dataset. Used to initialize the storage provider.
            mode (str): Mode in which the dataset is opened.
                Supported modes include ("r", "w", "a") plus an optional "+" suffix.
                Defaults to "a".
            provider (StorageProvider, optional): The storage provider used to access
                the data stored by this dataset.
            index: The Index object restricting the view of this dataset's tensors.
                Can be an int, slice, or (used internally) an Index object.

        Raises:
            ValueError: If an existing local path is given, it must be a directory.
            UserWarning: Both a path and provider should not be given.
        """
        self.mode = mode
        self.index = Index(index)

        if provider is not None and path:
            warnings.warn(
                "Dataset should not be constructed with both provider and path."
            )
        self.provider = provider or provider_from_path(path)

        self.tensors: Dict[str, Tensor] = {}
        if META_FILENAME in self.provider:
            ds_meta = read_dataset_meta(self.provider)
            for tensor_name in ds_meta["tensors"]:
                self.tensors[tensor_name] = Tensor(tensor_name, self.provider)
        else:
            write_dataset_meta(self.provider, {"tensors": []})

    def __len__(self):
        """Return the greatest length of tensors"""
        return max(map(len, self.tensors.values()), default=0)

    def __getitem__(self, item: Union[str, int, slice, Index]):
        if isinstance(item, str):
            if item not in self.tensors:
                raise TensorNotFoundError(item)
            else:
                return self.tensors[item][self.index]
        elif isinstance(item, (int, slice, Index)):
            new_index = self.index[Index(item)]
            return Dataset(mode=self.mode, provider=self.provider, index=new_index)
        else:
            raise InvalidKeyTypeError(item)

    def __setitem__(self, item: Union[slice, str], value):
        if isinstance(item, str):
            if isinstance(value, np.ndarray):
                write_array(
                    value,
                    item,
                    storage=self.provider,
                    batched=True,
                )
                ds_meta = read_dataset_meta(self.provider)
                ds_meta["tensors"].append(item)
                write_dataset_meta(self.provider, ds_meta)
                self.tensors[item] = Tensor(item, self.provider)
                return self.tensors[item]
            else:
                raise UnsupportedTensorTypeError(item)
        else:
            raise InvalidKeyTypeError(item)

    __getattr__ = __getitem__

    def __setattr__(self, name: str, value):
        """Set the named attribute on the dataset"""
        if name in DATASET_RESERVED_ATTRIBUTES:
            return super().__setattr__(name, value)
        else:
            return self.__setitem__(name, value)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

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
