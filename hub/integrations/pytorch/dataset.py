from hub.util.iterable_ordered_dict import IterableOrderedDict
from typing import Callable, Optional, Sequence, List
from hub.core.io import (
    SampleStreaming,
    Schedule,
    ShufflingSchedulerWrapper,
    SingleThreadScheduler,
    MultiThreadedNativeScheduler,
)
import torch
import numpy as np


def use_scheduler(num_workers: int):
    if num_workers <= 1:
        return SingleThreadScheduler()
    else:
        return MultiThreadedNativeScheduler(num_workers)


class TorchDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        dataset,
        use_local_cache: bool = False,
        tensors: Optional[Sequence[str]] = None,
        transform: Optional[Callable] = None,
        num_workers: int = 1,
        shuffle: bool = False,
    ) -> None:
        super().__init__()

        self.dataset = dataset
        self.transform = transform
        self.tensors = tensors

        self.scheduler = use_scheduler(num_workers)

        if shuffle:
            self.scheduler = ShufflingSchedulerWrapper(self.scheduler)

        streaming = SampleStreaming(
            dataset,
            scheduler=self.scheduler,
            tensors=self.tensors,
            use_local_cache=use_local_cache,
        )

        self.schedules: List[Schedule] = self.scheduler.schedule(
            streaming.list_blocks()
        )

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        schedule: Schedule = self.schedules[0]

        if worker_info is not None:
            schedule = self.schedules[worker_info.id]

        streaming = SampleStreaming(
            self.dataset,
            scheduler=self.scheduler,
            tensors=self.tensors,
            use_local_cache=False,
        )

        stream = streaming.read(schedule)

        yield from TorchDataset.apply_transform(
            stream=TorchDataset.map_to_tensor(self.to_pytensor, stream),
            transform=self.transform,
        )

    @staticmethod
    def apply_transform(stream, transform):
        if transform is not None:
            return map(transform, stream)
        else:
            return stream

    @staticmethod
    def map_to_tensor(map_func: Callable, stream):
        return map(
            lambda d: IterableOrderedDict((k, map_func(v)) for k, v in d.items()),
            stream,
        )

    def _remap_dtype(self, tensor):
        if tensor.dtype == np.uint16:
            return tensor.astype(np.int16)
        if tensor.dtype == np.uint32:
            return tensor.astype(np.int32)
        if tensor.dtype == np.uint64:
            return tensor.astype(np.int64)

        return tensor

    def to_pytensor(self, tensor):
        return torch.from_numpy(self._remap_dtype(tensor))
