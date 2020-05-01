from typing import *
import json, os, sys, time, random, uuid, io

from .array import Array, Props
from .storage import Base as Storage
from .dataset import Dataset, DatasetProps

try:
    import torch
except ImportError:
    pass

import numpy as np


class Bucket:
    _storage: Storage = None

    def __init__(self, storage: Storage):
        self._storage = storage

    def array(
        self,
        name: str,
        shape: Iterable[int],
        chunk: Iterable[int],
        dtype: Union[str, np.dtype],
        compress: str = "default",
        compresslevel: float = 0.5,
        dsplit: int = None,
    ) -> Array:

        return self.array_create(
            name, shape, chunk, dtype, compress, compresslevel, dsplit
        )

    def dataset(self, name: str, components: Dict[str, str]):
        return self.dataset_create(name, components)

    def open(self, name: str):
        jsontext = self._storage.get_or_none(os.path.join(name, "info.json"))
        if jsontext is None:
            return self.blob_get(name)
        else:
            d = json.loads(jsontext.decode())
            if "compresslevel" in d:
                return self.array_open(name)
            else:
                return self.dataset_open(name)

    def delete(self, name: str):
        jsontext = self._storage.get_or_none(os.path.join(name, "info.json"))
        if jsontext is None:
            if self._storage.get_or_none(name) is not None:
                self.blob_delete(name)
            else:
                raise Exception(f"Path f{name} is invalid for deletion")
        else:
            d = json.loads(jsontext.decode())
            if "compresslevel" in d:
                self.array_delete(name)
            else:
                self.dataset_delete(name)

    def array_create(
        self,
        name: str,
        shape: Iterable[int],
        chunk: Iterable[int],
        dtype: Union[str, np.dtype],
        compress: str = "default",
        compresslevel: float = 0.5,
        dsplit: int = None,
    ) -> Array:

        shape = tuple(shape)
        chunk = tuple(chunk)

        overwrite = False
        props = Props()
        props.shape = shape
        props.chunk = chunk
        props.dtype = np.dtype(dtype).str
        props.compress = compress
        props.compresslevel = compresslevel
        if dsplit is not None:
            assert isinstance(dsplit, int)
            props.darray = "darray"
            darray_path = os.path.join(name, props.darray)
            darray_shape = shape[:dsplit] + (len(shape) - dsplit,)
            arr = self.array_create(darray_path, darray_shape, darray_shape, "int32")
            slices = tuple(map(lambda s: slice(0, s), shape[:dsplit]))
            arr[slices] = shape[dsplit:]

        assert len(shape) == len(chunk)

        info_path = os.path.join(name, "info.json")
        chunk_path = os.path.join(name, "chunks")

        if overwrite or not self._storage.exists(info_path):
            self._storage.put(info_path, bytes(json.dumps(props.__dict__), "utf-8"))

        if overwrite and self._storage.exists(chunk_path):
            self._storage.delete(chunk_path)

        return Array(name, self._storage)

    def array_open(self, name: str) -> Array:
        return Array(name, self._storage)

    def array_delete(self, name: str):
        self._storage.delete(name)

    def dataset_create(self, name: str, components: Dict[str, str]) -> Dataset:
        overwrite = False
        props = DatasetProps()
        props.paths = {}
        for key, comp in components.items():
            if isinstance(comp, Array):
                props.paths[key] = comp._path
            elif isinstance(comp, str):
                props.paths[key] = comp
            else:
                raise Exception(
                    "Input to the dataset is unknown: {}:{}".format(k, value)
                )

        dataset_path = os.path.join(name, "info.json")

        if overwrite or not self._storage.exists(dataset_path):
            self._storage.put(dataset_path, bytes(json.dumps(props.__dict__), "utf-8"))

        return Dataset(name, self._storage)

    def dataset_open(self, name: str) -> Dataset:
        return Dataset(name, self._storage)

    def dataset_delete(self, name: str):
        return self._storage.delete(name)

    def blob_get(self, name: str, none_on_errors: bool = False) -> bytes:
        if none_on_errors:
            return self._storage.get_or_none(name)
        else:
            return self._storage.get(name)

    def blob_set(self, name: str, content: bytes):
        self._storage.put(name, content)

    def blob_delete(self, name: str) -> bytes:
        self._storage.delete(name)

    def pytorch_model_get(self, name: str, none_on_errors: bool = False) -> dict:
        blob = self.blob_get(name, none_on_errors)

        if blob is None and not none_on_errors:
            raise Error("Error loading pytorch model")
        elif blob is None:
            return None
        else:
            f = io.BytesIO(blob)
            return torch.load(f)

    def pytorch_model_set(self, name: str, model: dict):
        f = io.BytesIO()
        torch.save(model, f)
        f.seek(0)
        self.blob_set(name, f.read())

    def pytorch_model_delete(self, name: str):
        self.blob_delete(name)
