from deeplake.util.exceptions import TensorDoesNotExistError
from deeplake.core.linked_sample import LinkedSample
from deeplake.core.sample import Sample
from deeplake.core.tensor import Tensor
from deeplake.constants import MB

import numpy as np

import posixpath


class TransformTensor:
    def __init__(self, dataset, name, is_group=False):
        self.items = []
        self.dataset = dataset
        self.name = name
        self.is_group = is_group
        self.idx = slice(None, None, None)

    def __len__(self):
        return len(self.items)

    def __getattr__(self, item):
        del self.dataset.data[self.name]
        return self.dataset[posixpath.join(self.name, item)][self.idx]

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.__getattr__(item)
        self.idx = item
        return self

    def numpy(self):
        if isinstance(self.idx, int):
            items = [self.numpy_compressed()]
            squeeze = True
        else:
            items = self.numpy_compressed()
            squeeze = False

        values = []
        for i, item in enumerate(items):
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
        return self.items[self.idx]

    def append(self, item):
        if self.is_group:
            raise TensorDoesNotExistError(self.name)
        if self.dataset.all_chunk_engines:
            self.dataset.item_added(item)
        self.items.append(item)

    def extend(self, items):
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
        cache_size=16 * MB,
    ):
        self.tensors = tensors
        self.data = {tensor: TransformTensor(self, tensor) for tensor in tensors}
        self.all_chunk_engines = all_chunk_engines
        self.group_index = group_index
        self.label_temp_tensors = label_temp_tensors
        self.cache_size = cache_size
        self.cache_used = 0
        self.idx = idx

    def __len__(self):
        return min(len(self[tensor]) for tensor in self.data)

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
            sizeof_item = np.asarray(item).nbytes

        self.cache_used += sizeof_item
        if self.cache_used >= self.cache_size:
            self.flush()

    def flush(self):
        all_chunk_engines = self.all_chunk_engines
        label_temp_tensors = self.label_temp_tensors
        for name, tensor in self.data.items():
            if not tensor.is_group:
                name = posixpath.join(self.group_index, name)
                chunk_engine = all_chunk_engines[label_temp_tensors.get(name) or name]
                callback = chunk_engine._transform_callback
                chunk_engine.extend(
                    tensor[:].numpy_compressed(), link_callback=callback
                )
                tensor.items.clear()


# class TransformDataset:
#     def __init__(self, all_tensors=None, slice_list=None):
#         """Creates a Dataset like object that supports "." access of tensors and appends/extends to the tensors.
#         This is used as sample_out in deeplake transforms.
#         """
#         self.tensors = all_tensors or {}
#         self.slice_list = slice_list or []

#     def __len__(self):
#         return min(len(self[tensor]) for tensor in self.tensors)

#     def __getattr__(self, name):
#         if name not in self.tensors:
#             self.tensors[name] = TransformTensor(name=name, dataset=self)
#         return self.tensors[name][self.slice_list]

#     def __getitem__(self, slice_):
#         if isinstance(slice_, str):
#             return self.__getattr__(slice_)
#         assert isinstance(slice_, (slice, int))
#         new_slice_list = self.slice_list + [slice_]
#         return TransformDataset(all_tensors=self.tensors, slice_list=new_slice_list)

#     def __iter__(self):
#         for i in range(len(self)):
#             yield self[i]

#     def append(self, sample, skip_ok=False):
#         if not skip_ok:
#             for k in self.tensors:
#                 if k not in sample:
#                     raise TensorDoesNotExistError(k)
#         if len(set(map(len, (self[k] for k in sample)))) != 1:
#             raise ValueError("All tensors are expected to have the same length.")
#         for k, v in sample.items():
#             self[k].append(v)
