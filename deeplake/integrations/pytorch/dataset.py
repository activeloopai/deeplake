from typing import Optional, Sequence, List, Dict
from deeplake.constants import MB
from deeplake.integrations.pytorch.common import PytorchTransformFunction
from deeplake.util.exceptions import TransformFailedError

from deeplake.util.iterable_ordered_dict import IterableOrderedDict
from deeplake.util.warnings import always_warn
from deeplake.core.io import (
    DistributedScheduler,
    SampleStreaming,
    Schedule,
    SequentialMultithreadScheduler,
    ShufflingSchedulerWrapper,
    SingleThreadScheduler,
    MultiThreadedNaiveScheduler,
)
from deeplake.core.sample import Sample
from deeplake.core.polygon import Polygons
from deeplake.integrations.pytorch.shuffle_buffer import ShuffleBuffer

import torch
import torch.utils.data
import torch.distributed as dist
from torch.utils.data.dataloader import DataLoader
from warnings import warn

import numpy as np
from PIL import Image  # type: ignore


mp = torch.multiprocessing.get_context()


def identity(x):
    return x


def use_scheduler(num_workers: int, ensure_order: bool, batch_size: int = 1):
    if num_workers <= 1:
        return SingleThreadScheduler()
    else:
        if ensure_order:
            return MultiThreadedNaiveScheduler(num_workers)
        else:
            return SequentialMultithreadScheduler(num_workers, batch_size)


def cast_type(tensor):
    if isinstance(tensor, list):
        return tensor
    # Cast to a pytorch supported dtype.
    if tensor.dtype == np.uint16:
        return tensor.astype(np.int32)
    if tensor.dtype == np.uint32:
        return tensor.astype(np.int64)
    if tensor.dtype == np.uint64:
        return tensor.astype(np.int64)
    return None  # if not casted, calling method might want to make a copy.


def copy_tensor(x):
    if isinstance(x, dict):
        return x.copy()
    if isinstance(x, Sample):
        x = x.array
    if isinstance(x, Image.Image):
        return x
    if isinstance(x, str):
        return x

    try:
        copy = cast_type(x)
    except AttributeError:
        return bytes(x)
    if copy is None:
        copy = x.copy()
    if isinstance(copy, Polygons):
        copy = copy.numpy()
    return copy


def _process(sample, transform: Optional[PytorchTransformFunction], return_index: bool):
    sample = IterableOrderedDict((k, copy_tensor(sample[k])) for k in sample.keys())
    index = sample["index"][0]
    if not return_index:
        del sample["index"]
    if transform:
        try:
            return transform(sample)
        except Exception as e:
            raise TransformFailedError(index) from e
    return sample


class TorchDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        dataset,
        tensors: Sequence[str],
        use_local_cache: bool = False,
        transform: Optional[PytorchTransformFunction] = PytorchTransformFunction(),
        num_workers: int = 1,
        shuffle: bool = False,
        buffer_size: int = 0,
        return_index: bool = True,
        pad_tensors: bool = False,
        decode_method: Optional[Dict[str, str]] = None,
        batch_size: int = 1,
        cache_size: int = 32 * MB,
    ) -> None:
        super().__init__()

        self.dataset = dataset
        self.transform = transform
        self.tensors = tensors
        self.shuffle: bool = shuffle
        self.buffer_size: int = buffer_size * MB
        self.return_index: bool = return_index
        self.pad_tensors = pad_tensors
        self.decode_method = decode_method
        self.batch_size = batch_size
        self.cache_size = cache_size

        self.use_local_cache = use_local_cache
        self.scheduler = use_scheduler(num_workers, shuffle, batch_size)

        if dist.is_initialized():
            self.scheduler = DistributedScheduler(num_workers)

        if shuffle:
            self.scheduler = ShufflingSchedulerWrapper(self.scheduler)

        streaming = SampleStreaming(
            dataset,
            tensors=self.tensors,  # type: ignore
            use_local_cache=use_local_cache,
            pad_tensors=self.pad_tensors,
            decode_method=self.decode_method,
            verbose=False,
            cache_size=cache_size,
        )

        self.schedules: List[Schedule] = self.scheduler.schedule(
            streaming.list_blocks()
        )

    def __iter__(self: "TorchDataset"):
        worker_info = torch.utils.data.get_worker_info()
        schedule: Schedule = self.schedules[0]

        if worker_info is not None:
            schedule = self.schedules[worker_info.id]

        streaming = SampleStreaming(
            self.dataset,
            tensors=self.tensors,
            use_local_cache=self.use_local_cache,
            pad_tensors=self.pad_tensors,
            decode_method=self.decode_method,
        )

        if self.shuffle:
            schedule.shuffle()

        stream = streaming.read(schedule)

        for data in stream:
            yield _process(data, self.transform, self.return_index)

    def __len__(self):
        return sum(map(len, self.schedules))


class SubIterableDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        dataset,
        tensors: Sequence[str],
        use_local_cache: bool = False,
        transform: PytorchTransformFunction = PytorchTransformFunction(),
        num_workers: int = 1,
        buffer_size: int = 512,
        batch_size: int = 1,
        return_index: bool = True,
        pad_tensors: bool = False,
        decode_method: Optional[Dict[str, str]] = None,
        cache_size: int = 32 * MB,
    ) -> None:
        super().__init__()

        self.torch_datset = TorchDataset(
            dataset,
            tensors=tensors,
            use_local_cache=use_local_cache,
            transform=transform,
            num_workers=num_workers,
            shuffle=True,
            return_index=return_index,
            pad_tensors=pad_tensors,
            decode_method=decode_method,
            cache_size=cache_size,
        )
        if buffer_size:
            self.transform = transform
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.buffer_size = buffer_size * MB
        self.return_index = return_index
        if self.buffer_size == 0:
            warn("setting buffer_size = 0 will result in poor shuffling distribution")

    def __iter__(self):
        sub_loader = DataLoader(
            self.torch_datset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=identity,
        )
        buffer_size = self.buffer_size
        if buffer_size:
            buffer = ShuffleBuffer(buffer_size)
            it = iter(sub_loader)
            try:
                while True:
                    next_batch = next(it)
                    for val in next_batch:
                        result = buffer.exchange(val)
                        if result is not None:
                            yield result
                    del next_batch
            except StopIteration:
                pass
            while not buffer.emtpy():
                yield buffer.exchange(None)
            del it
        else:
            for batch in sub_loader:
                yield from batch
        del sub_loader

    def __len__(self):
        return len(self.torch_datset)
