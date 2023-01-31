from deeplake.util.exceptions import TensorDoesNotExistError
from deeplake.core.linked_sample import LinkedSample
from deeplake.core.sample import Sample
from deeplake.core.tensor import Tensor
from typing import Union, List, Any
from deeplake.constants import MB
from itertools import chain

import numpy as np

import posixpath
import bisect


class TransformTensor:
    def __init__(self, dataset, name, is_group=False):
        self.items = []
        self.dataset = dataset
        self.name = name
        self.is_group = is_group
        self.idx = slice(None, None, None)
        self.numpy_only = True
        self.cum_sizes = []

    def __len__(self):
        if self.numpy_only:
            return 0 if not self.cum_sizes else self.cum_sizes[-1]
        return len(self.items)

    def __getattr__(self, item):
        return self.dataset[posixpath.join(self.name, item)][self.idx]

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.__getattr__(item)
        self.idx = item
        return self

    def numpy(self) -> Union[List, np.ndarray]:
        if self.numpy_only:
            return self.numpy_compressed()

        if isinstance(self.idx, int):
            items = [self.numpy_compressed()]
            squeeze = True
        else:
            items = self.numpy_compressed()
            squeeze = False

        values: List[Any] = []
        for item in items:
            if isinstance(item, Sample):
                values.append(item.array)
            elif not isinstance(item, (LinkedSample, Tensor, type(None))):
                values.append(np.asarray(item))
            else:
                values.append(item)
        if squeeze:
            values = values[0]
        return values

    def numpy_compressed(self):
        idx = self.idx
        if self.numpy_only:
            if isinstance(idx, int):
                i = bisect.bisect_right(self.cum_sizes, idx)
                if i == 0:
                    j = idx
                else:
                    j = idx - self.cum_sizes[i - 1]
                return self.items[i][j]
        return self.items[idx]

    def non_numpy_only(self):
        if self.numpy_only:
            items = list(chain(*self.items[:]))
            self.items.clear()
            self.items += items
            self.cum_sizes.clear()
            self.numpy_only = False

    def append(self, item):
        if self.is_group:
            raise TensorDoesNotExistError(self.name)
        if self.numpy_only:
            # optimization applicable only if extending
            self.non_numpy_only()
        self.items.append(item)
        if self.dataset.all_chunk_engines:
            self.dataset.item_added(item)

    def extend(self, items):
        if self.numpy_only:
            if isinstance(items, np.ndarray):
                self.items.append(items)
                if len(self.cum_sizes) == 0:
                    self.cum_sizes.append(len(items))
                else:
                    self.cum_sizes.append(self.cum_sizes[-1] + len(items))
                if self.dataset.all_chunk_engines:
                    self.dataset.item_added(items)
                return
            else:
                self.non_numpy_only()

        for item in items:
            self.append(item)


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
        elif isinstance(item, (Tensor, type(None))):
            sizeof_item = 0
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
