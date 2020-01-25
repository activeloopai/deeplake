from typing import Tuple, Dict, Optional
import json

from .array import Array, Props
from .storage import Base as Storage
from .dataset import Dataset, DatasetProps


class Bucket():
    _storage: Storage = None

    def __init__(self, storage: Storage):
        self._storage = storage

    def array_create(
        self, 
        name: str, 
        shape: Tuple[int, ...], 
        chunk: Tuple[int, ...], 
        dtype: str, 
        compress: str = 'default', 
        compresslevel: float = 0.5, 
        ) -> Array:
        
        overwrite = False
        props = Props()
        props.shape = shape
        props.chunk = chunk
        props.dtype = dtype
        props.compress = compress
        props.compresslevel = compresslevel

        assert len(shape) == len(chunk)
        
        if overwrite or not self._storage.exists(name + '/info.json'):
            self._storage.put(name + '/info.json', bytes(json.dumps(props.__dict__), 'utf-8'))

        if overwrite and self._storage.exists(name + '/chunks'):
            self._storage.delete(name + '/chunks')

        return Array(name, self._storage)

    def array_open(self, name: str) -> Array:
        return Array(name, self._storage)

    def array_delete(self, name: str):
        return self._storage.delete(name)

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
                    'Input to the dataset is unknown: {}:{}'.format(k, value))
        
        if overwrite or not self._storage.exists(name + '/info.json'):
            self._storage.put(name + '/info.json', bytes(json.dumps(props.__dict__), 'utf-8'))
        
        return Dataset(name, self._storage)

    def dataset_open(self, name: str) -> Dataset:
        return Dataset(name, self._storage)

    def dataset_delete(self, name: str):
        return self._storage.delete(name) 