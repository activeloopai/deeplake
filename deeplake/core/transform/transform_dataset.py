from deeplake.core.transform.transform_tensor import TransformTensor
from deeplake.core.linked_tiled_sample import LinkedTiledSample
from deeplake.core.partial_sample import PartialSample
from deeplake.core.linked_sample import LinkedSample
from deeplake.core.sample import Sample
from deeplake.core.tensor import Tensor
from deeplake.constants import MB


import numpy as np

import posixpath


class TransformDataset:
    def __init__(
        self,
        tensors,
        all_chunk_engines=None,
        group_index="",
        label_temp_tensors=None,
        idx=slice(None, None, None),
        cache_size=16,
    ):
        self.tensors = tensors
        self.data = {tensor: TransformTensor(self, tensor) for tensor in tensors}
        self.all_chunk_engines = all_chunk_engines
        self.group_index = group_index
        self.label_temp_tensors = label_temp_tensors
        self.cache_size = cache_size * MB
        self.cache_used = 0
        self.idx = idx
        self.pg_callback = None

    def __len__(self):
        return max(len(self[tensor]) for tensor in self.data)

    def __getattr__(self, tensor):
        try:
            return self.data[tensor][self.idx]
        except KeyError:
            self.data[tensor] = TransformTensor(self, tensor, is_group=True)
            return self.data[tensor][self.idx]

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.__getattr__(item)
        assert isinstance(item, (slice, int))
        self.idx = item
        return self

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def append(self, sample):
        if len(set(map(len, (self[k] for k in sample)))) != 1:
            raise ValueError("All tensors are expected to have the same length.")
        for k, v in sample.items():
            self[k].append(v)

    def item_added(self, item):
        if isinstance(item, Sample):
            sizeof_item = len(item.buffer)
        elif isinstance(item, LinkedSample):
            sizeof_item = len(item.path)
        elif isinstance(item, np.ndarray):
            sizeof_item = item.nbytes
        elif isinstance(item, (Tensor, type(None), PartialSample)):
            sizeof_item = 0
        elif isinstance(item, LinkedTiledSample):
            sizeof_item = item.path_array.nbytes
        else:
            sizeof_item = np.asarray(item, dtype=object).nbytes

        self.cache_used += sizeof_item
        if self.cache_used >= self.cache_size:
            self.flush()

    def set_pg_callback(self, callback):
        self.pg_callback = callback

    def flush(self):
        all_chunk_engines = self.all_chunk_engines
        label_temp_tensors = self.label_temp_tensors
        for name, tensor in self.data.items():
            if not tensor.is_group:
                name = posixpath.join(self.group_index, name)
                chunk_engine = all_chunk_engines[label_temp_tensors.get(name, name)]
                callback = chunk_engine._transform_callback
                if tensor.numpy_only:
                    items = tensor[:].numpy_compressed()
                    for item in items:
                        chunk_engine.extend(
                            item, link_callback=callback, pg_callback=self.pg_callback
                        )
                else:
                    chunk_engine.extend(
                        tensor[:].numpy_compressed(),
                        link_callback=callback,
                        pg_callback=self.pg_callback,
                    )
                tensor.items.clear()
        self.cache_used = 0
