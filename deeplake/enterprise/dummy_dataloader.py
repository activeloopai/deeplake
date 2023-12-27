from typing import Union
from deeplake.core.compression import compress_array
from deeplake.util.exceptions import TensorDoesNotExistError
from deeplake.util.iterable_ordered_dict import IterableOrderedDict
from deeplake.util.keys import get_sample_shape_tensor_key
import numpy as np
from PIL import Image  # type: ignore


def identity(x):
    return x


def get_mode(tensor, raw_tensors, pil_compressed_tensors):
    if tensor in pil_compressed_tensors:
        mode = "pil"
    elif tensor in raw_tensors:
        mode = "raw"
    else:
        mode = "numpy"
    return mode


def upcast_array(arr: Union[np.ndarray, bytes]):
    if isinstance(arr, list):
        return [upcast_array(a) for a in arr]
    if isinstance(arr, np.ndarray):
        if arr.dtype == np.uint16:
            return arr.astype(np.int32)
        if arr.dtype == np.uint32:
            return arr.astype(np.int64)
        if arr.dtype == np.uint64:
            return arr.astype(np.int64)
    return arr


class DummyTensor:
    def __init__(self, dataset, tensor_name, mode):
        tensor = dataset[tensor_name]
        shape_tensor_key = get_sample_shape_tensor_key(tensor_name)
        self.dtype = tensor.dtype
        self.htype = tensor.htype
        self.sample_compression = tensor.meta.sample_compression
        max_shape, min_shape = tensor.meta.max_shape, tensor.meta.min_shape
        print(f"Fetching shape data for {tensor_name}")
        if max_shape == min_shape:
            print(
                f"Shape data for {tensor_name} is constant. Using {max_shape} for all samples."
            )
            self.tensor_shapes = [max_shape] * len(dataset)
        else:
            try:
                self.tensor_shapes = dataset[shape_tensor_key].numpy(
                    aslist=True, fetch_chunks=True
                )
            except TensorDoesNotExistError:
                print(
                    f"Shape tensor {shape_tensor_key} does not exist. Generating random shapes."
                )
                self.tensor_shapes = []
                for _ in range(len(dataset)):
                    shape = [
                        np.random.randint(min_dim, max_dim + 1)
                        for min_dim, max_dim in zip(min_shape, max_shape)
                    ]
                    self.tensor_shapes.append(tuple(shape))

        self.mode = mode

    def get_numpy_data(self, index):
        dtype = self.dtype
        shape = self.tensor_shapes[index]
        if len(shape) == 0:
            shape = (1,)
        if dtype == "str":
            if 0 in shape:
                shape = (1,)
            data = np.array(["a" * np.prod(shape)], dtype=dtype)
            # handle json, list separately later
        else:
            data = np.ones(shape, dtype=dtype)
        if self.htype == "polygon":
            data = list(data)
        return data

    def get_data(self, index):
        data = self.get_numpy_data(index)
        mode, compression = self.mode, self.sample_compression
        if mode == "numpy":
            return data
        elif mode == "raw":
            return compress_array(data, compression)
        elif self.mode == "pil":
            return Image.fromarray(data)

    def __getitem__(self, index):
        return self.get_data(index)


class DummyDataset:
    def __init__(
        self,
        deeplake_dataset,
        tensors,
        transform_fn,
        upcast,
        return_index,
        raw_tensors,
        pil_compressed_tensors,
    ):
        self.tensors = {}
        self.length = len(deeplake_dataset)
        for tensor in tensors:
            mode = get_mode(tensor, raw_tensors, pil_compressed_tensors)
            self.tensors[tensor] = DummyTensor(deeplake_dataset, tensor, mode)
        self.upcast = upcast
        self.return_index = return_index
        self.transform_fn = transform_fn

    def __getitem__(self, index):
        sample = IterableOrderedDict(
            {tensor: tensor_obj[index] for tensor, tensor_obj in self.tensors.items()}
        )
        if self.return_index:
            sample["index"] = np.array([index])
        if self.upcast:
            sample = IterableOrderedDict(
                (k, upcast_array(v)) for k, v in sample.items()
            )
        if self.transform_fn:
            sample = self.transform_fn(sample)
        return sample

    def __len__(self):
        return self.length


class DummyDataloader:
    def __init__(
        self,
        deeplake_dataset,
        batch_size,
        shuffle,
        num_workers,
        collate_fn,
        transform_fn,
        distributed,
        prefetch_factor,
        tensors,
        drop_last,
        upcast,
        return_index,
        raw_tensors,
        pil_compressed_tensors,
        persistent_workers,
    ):
        import torch
        from torch.utils.data.distributed import DistributedSampler

        self.dataset = DummyDataset(
            deeplake_dataset,
            tensors,
            transform_fn,
            upcast,
            return_index,
            raw_tensors,
            pil_compressed_tensors,
        )
        sampler = DistributedSampler(self.dataset) if distributed else None
        prefetch_factor = prefetch_factor if num_workers and num_workers > 0 else 2
        kwargs = {
            "batch_size": batch_size or 1,
            "shuffle": shuffle or False,
            "num_workers": num_workers or 0,
            "collate_fn": collate_fn or identity,
            "sampler": sampler,
            "drop_last": drop_last or False,
        }
        if num_workers and num_workers > 0:
            kwargs["prefetch_factor"] = prefetch_factor
            kwargs["persistent_workers"] = persistent_workers
        self.loader = torch.utils.data.DataLoader(self.dataset, **kwargs)

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)

    @property
    def summary(self):
        return
