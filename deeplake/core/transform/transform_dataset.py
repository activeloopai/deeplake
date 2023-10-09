from deeplake.util.exceptions import (
    SampleAppendError,
    SampleAppendingError,
    SampleExtendingError,
    TensorDtypeMismatchError,
)
from deeplake.core.transform.transform_tensor import TransformTensor
from deeplake.util.json import validate_json_object, HubJsonEncoder
from deeplake.core.linked_tiled_sample import LinkedTiledSample
from deeplake.core.partial_sample import PartialSample
from deeplake.core.linked_sample import LinkedSample
from deeplake.util.casting import intelligent_cast
from deeplake.core.sample import Sample
from deeplake.core.tensor import Tensor
from deeplake.constants import MB


import numpy as np

import posixpath
import json


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
        self.start_input_idx = None

    def set_start_input_idx(self, start_input_idx):
        if self.start_input_idx is None:
            self.start_input_idx = start_input_idx

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

    def _get_engine_name(self, name):
        name = posixpath.join(self.group_index, name)
        name = self.label_temp_tensors.get(name, name)
        return name

    def append(self, sample, skip_ok=False, append_empty=False):
        if not isinstance(sample, dict):
            raise SampleAppendingError()

        if skip_ok:
            raise ValueError(
                "`skip_ok` is not supported for `ds.append` in transforms. Use `skip_ok` parameter of the `eval` method instead."
            )

        if len(set(map(len, (self[k] for k in sample)))) != 1:
            raise ValueError(
                "All tensors are expected to have the same length before `ds.append`."
            )

        for k in self.tensors:
            if k in sample:
                self[k].append(sample[k])
            elif append_empty:
                self[k].append(None)

    def extend(self, sample, skip_ok=False, append_empty=False):
        if not isinstance(sample, dict):
            raise SampleExtendingError()

        if skip_ok:
            raise ValueError(
                "`skip_ok` is not supported for `ds.extend` in transforms. Use `skip_ok` parameter of the `eval` method instead."
            )

        if len(set(map(len, (self[k] for k in sample)))) != 1:
            raise ValueError(
                "All tensors are expected to have the same length before `ds.extend`."
            )

        n = len(next(iter(sample.values())))
        for v in sample.values():
            if len(v) != n:
                sizes = {k: len(v) for (k, v) in sample.items()}
                raise ValueError(
                    f"Incoming samples are not of equal lengths. Incoming sample sizes: {sizes}"
                )

        for i in range(n):
            self.append(
                {k: v[i] for (k, v) in sample.items()}, append_empty=append_empty
            )

    def update(self, sample):
        raise NotImplementedError("ds.update is not supported in transforms.")

    def _calculate_sample_size(self, item, dtype, htype):
        if isinstance(item, str):
            return len(item.encode())
        elif htype in ("json", "list"):
            # NOTE: These samples will be serialized twice. Once here, and once in the chunk engine.
            validate_json_object(item, dtype)
            byts = json.dumps(item, cls=HubJsonEncoder).encode()
            return len(byts)
        else:
            try:
                item = intelligent_cast(item, dtype, htype)
            except TensorDtypeMismatchError:
                # class_label tensor can have integer dtype, but sample can be a string
                if htype == "class_label":
                    item = intelligent_cast(item, "str", htype)
                else:
                    raise
            return item.nbytes

    def item_added(self, item, tensor):
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
            try:
                name = self._get_engine_name(tensor)
                chunk_engine = self.all_chunk_engines[name]
                meta = chunk_engine.tensor_meta
                htype, dtype = meta.htype, meta.dtype
                # First sample in tensor
                # Flush to set meta attributes
                if dtype is None:
                    self.flush()
                    return
                sizeof_item = self._calculate_sample_size(item, dtype, htype)
            except:
                sizeof_item = 0

        self.cache_used += sizeof_item

    def set_pg_callback(self, callback):
        self.pg_callback = callback

    def check_flush(self):
        if self.cache_used >= self.cache_size:
            self.flush()

    def _flush_numpy_tensor_to_chunk_engine(
        self, full_name, tensor, chunk_engine, callback, updated_tensors
    ):
        items = tensor[:].numpy_compressed()
        for item in items:
            chunk_engine.extend(
                item,
                link_callback=callback,
                pg_callback=self.pg_callback,
            )
            updated_tensors[full_name] += len(item)
        tensor.items.clear()

    def _flush_tensor_to_chunk_engine(
        self, full_name, tensor, chunk_engine, callback, updated_tensors
    ):
        items = tensor[:].numpy_compressed()
        chunk_engine.extend(
            items,
            link_callback=callback,
            pg_callback=self.pg_callback,
        )
        updated_tensors[full_name] = len(items)
        tensor.items.clear()

    def _rollback(self, updated_tensors):
        for t in updated_tensors:
            chunk_engine = self.all_chunk_engines[t]
            num_samples = updated_tensors[t]
            for _ in range(num_samples):
                chunk_engine.pop(link_callback=chunk_engine._transform_pop_callback)

    def _clear(self):
        for tensor in self.data.values():
            tensor.items.clear()
        self.cache_used = 0

    def flush(self):
        all_chunk_engines = self.all_chunk_engines
        label_temp_tensors = self.label_temp_tensors
        updated_tensors = {}
        try:
            for name, tensor in self.data.items():
                if not tensor.is_group:
                    name = self._get_engine_name(name)
                    updated_tensors[name] = 0
                    chunk_engine = all_chunk_engines[name]
                    callback = chunk_engine._transform_callback

                    if tensor.numpy_only:
                        self._flush_numpy_tensor_to_chunk_engine(
                            name, tensor, chunk_engine, callback, updated_tensors
                        )
                    else:
                        self._flush_tensor_to_chunk_engine(
                            name, tensor, chunk_engine, callback, updated_tensors
                        )
            self.start_input_idx = None
        except Exception as e:
            self._rollback(updated_tensors)
            e = e.__cause__ if isinstance(e, SampleAppendError) else e  # type: ignore
            raise SampleAppendError(name) from e
        finally:
            self._clear()
