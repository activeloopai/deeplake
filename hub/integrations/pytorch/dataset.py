from typing import Callable, Iterable, Optional, Sequence, List, Union
from hub.util.iterable_ordered_dict import IterableOrderedDict
from hub.core.io import (
    SampleStreaming,
    Schedule,
    ShufflingSchedulerWrapper,
    SingleThreadScheduler,
    MultiThreadedNativeScheduler,
)
from hub.integrations.pytorch.shuffle_buffer import ShuffleBuffer
import torch
from torch.multiprocessing import Queue, Process
from torch._utils import ExceptionWrapper

import numpy as np
from warnings import warn
from queue import Empty


mp = torch.multiprocessing.get_context()


def use_scheduler(num_workers: int):
    if num_workers <= 1:
        return SingleThreadScheduler()
    else:
        return MultiThreadedNativeScheduler(num_workers)


def cast_type(tensor: np.array):
    if tensor.dtype == np.uint16:
        return tensor.astype(np.int32)
    if tensor.dtype == np.uint32:
        return tensor.astype(np.int64)
    if tensor.dtype == np.uint64:
        return tensor.astype(np.int64)

    return tensor


class _ContinueIteration:

    """
    Instructs worker to load N samples of data and publish it to the
    data_queue

    Args:
        n: int number of samples to publish. Acts as in fly buffer_size similar
           to reactive stream approach
    """

    def __init__(self, n: int) -> None:
        self.n = n


class _StopIteration:

    """
    Acts as two way communication class.

    1. If received by worker, instructs to publish None to data queue and terminate loop

    2. If received by parent, indicates worker was terminated with Exception

    Args:
        exception: Optional[Exception] optional value for case #2
    """

    def __init__(self, exception: Optional[ExceptionWrapper] = None) -> None:
        self.reason = exception


def _worker_loop(
    dataset,
    tensors,
    use_local_cache: bool,
    schedule: Schedule,
    transform: Callable,
    request_queue: Queue,
    data_queue: Queue,
    workers_done,
):
    streaming = SampleStreaming(
        dataset,
        tensors=tensors,
        use_local_cache=use_local_cache,
    )

    is_running = True
    schedule.shuffle()

    it = iter(streaming.read(schedule))  # data samples iterator for current worker
    requested: int = 0  # indicate value of samples requested from worker

    while is_running:
        try:
            # Check if parent iterator is terminating. If so, no reason to continue.
            # if workers_done.is_set():
            # is_running = False
            # break

            # publish requested data
            if requested > 0:
                try:
                    data = next(it, None)

                    if data:
                        data = IterableOrderedDict(
                            (k, cast_type(v)) for k, v in data.items()
                        )

                        if transform:
                            data = transform(data)

                        data = IterableOrderedDict(
                            (k, torch.as_tensor(v)) for k, v in data.items()
                        )
                        data_queue.put(data)

                        requested -= 1
                    else:
                        is_running = False
                        requested = 0
                        break
                except KeyboardInterrupt:
                    pass
                except Exception as e:
                    # propagate exception back to parent
                    data_queue.put(_StopIteration(ExceptionWrapper(e)))
                    is_running = False
                    break

            # process commands from parent
            r: Union[_ContinueIteration, _StopIteration]
            try:
                if requested > 0:
                    r = request_queue.get_nowait()
                else:
                    r = request_queue.get(timeout=0.05)
            except Empty:
                # TODO can do some prefetch at this point
                # however, blocking thread for long might make it unresponsive.

                # Just continue for now
                continue

            # Continue Iteration instruction
            if isinstance(r, _ContinueIteration):
                requested += r.n

            # Termination instruction
            if isinstance(r, _StopIteration):
                is_running = False

        except KeyboardInterrupt:
            pass

    # None indicates that worker have no more data
    data_queue.put(None)

    # clear request queue and wait for final None
    # to avoid zombie processes
    while True:
        try:
            data = request_queue.get()
            if not data:
                break
        except KeyboardInterrupt:
            pass

    # gracefully close connections
    request_queue.cancel_join_thread()
    request_queue.close()


class PrefetchConcurrentIterator(Iterable):

    """
    Prefetching iterator, that starts dataset.num_workers processes, that fetch data
    according to dataset.schedules[worker_id] and returning to single threaded
    dataset iterator.

    Iterator doesn't guarantee order of data, between threads, those ideal
    for shuffling dataset.
    """

    def __init__(self, dataset) -> None:
        super().__init__()
        self.dataset = dataset
        self.num_workers = dataset.num_workers
        self.buffer_size = dataset.buffer_size

        self.workers: List[Process] = []
        self.request_queues: List[Queue] = [Queue() for _ in range(self.num_workers)]
        self.data_queues: List[Queue] = [Queue() for _ in range(self.num_workers)]
        self.queue_size = [0 for _ in range(self.num_workers)]
        self.active_workers = [False for _ in range(self.num_workers)]
        self.workers_done = mp.Event()
        self.shutdown = False

        for i in range(self.num_workers):
            w = mp.Process(
                target=_worker_loop,
                args=(
                    dataset.dataset,
                    dataset.tensors,
                    dataset.use_local_cache,
                    dataset.schedules[i],
                    dataset.transform,
                    self.request_queues[i],
                    self.data_queues[i],
                    self.workers_done,
                ),
            )
            w.daemon = True
            w.start()

            self.workers.append(w)
            self.active_workers[i] = True
            self.request_queues[i].put_nowait(_ContinueIteration(dataset.queue_size))

    def __iter__(self):
        shuffle_buffer = ShuffleBuffer(self.buffer_size)

        while any(self.active_workers):
            for wid in range(self.num_workers):
                if self.active_workers[wid]:

                    queue = self.data_queues[wid]

                    # if low on messages in a queue, request more
                    if self.queue_size[wid] > 0:
                        self.request_queues[wid].put(
                            _ContinueIteration(self.queue_size[wid])
                        )
                        self.queue_size[wid] = 0

                    try:
                        data = queue.get(timeout=0.5)
                    except Empty:
                        if not self.workers[wid].is_alive():
                            self.active_workers[wid] = False
                            warn(f"worker id-{wid} terminated")

                        continue

                    if data:
                        # Handling emergency termination of sequence
                        if isinstance(data, _StopIteration):
                            if data.reason:
                                data.reason.reraise()

                        # else return sample
                        self.queue_size[wid] += 1

                        # exchange sample with shuffle buffer
                        if self.buffer_size > 0:
                            data = shuffle_buffer.exchange(data)

                            if data:
                                yield data
                        else:
                            yield data
                    else:
                        # Mark worker as done with streaming
                        self.active_workers[wid] = False

        self._shutdown_all()

        # clear all samples from shuffle buffer
        if self.buffer_size > 0:
            while len(shuffle_buffer) > 0:
                data = shuffle_buffer.exchange(None)
                yield data

    def __len__(self) -> int:
        return len(self.dataset)

    def __del__(self):
        self._shutdown_all()

    def _shutdown_all(self):
        if not self.shutdown:
            self.workers_done.set()

            try:
                for wid in range(self.num_workers):
                    self.request_queues[wid].put(_StopIteration())
                    self.request_queues[wid].put(None)
                    self.workers[wid].join(timeout=5)

                for queue in self.request_queues:
                    queue.cancel_join_thread()
                    queue.close()
            finally:
                for worker in self.workers:
                    if worker.is_alive():
                        worker.terminate()

            self.shutdown = True


class ShufflingIterableDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        dataset,
        use_local_cache: bool = False,
        tensors: Optional[Sequence[str]] = None,
        transform: Optional[Callable] = None,
        num_workers: int = 1,
        buffer_size: int = 0,
        queue_size: int = 64,
    ) -> None:

        super().__init__()

        assert num_workers >= 1

        self.dataset = dataset
        self.num_workers = num_workers
        self.transform = transform
        self.tensors = tensors
        self.use_local_cache = use_local_cache
        self.scheduler = ShufflingSchedulerWrapper(
            MultiThreadedNativeScheduler(self.num_workers)
        )

        self.queue_size = queue_size

        streaming = SampleStreaming(
            dataset,
            tensors=self.tensors,
            use_local_cache=use_local_cache,
        )

        self.schedules: List[Schedule] = self.scheduler.schedule(
            streaming.list_blocks()
        )

        self.buffer_size: Optional[int] = buffer_size

    def __iter__(self):
        it = PrefetchConcurrentIterator(self)
        return iter(it)

    def __len__(self):
        return sum(map(len, self.schedules))


class TorchDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        dataset,
        use_local_cache: bool = False,
        tensors: Optional[Sequence[str]] = None,
        transform: Optional[Callable] = None,
        num_workers: int = 1,
        shuffle: bool = False,
        buffer_size: int = 0,
    ) -> None:
        super().__init__()

        self.dataset = dataset
        self.transform = transform
        self.tensors = tensors
        self.use_local_cache = use_local_cache
        self.scheduler = use_scheduler(num_workers)

        if shuffle:
            self.scheduler = ShufflingSchedulerWrapper(self.scheduler)

        streaming = SampleStreaming(
            dataset,
            tensors=self.tensors,
            use_local_cache=use_local_cache,
        )

        self.schedules: List[Schedule] = self.scheduler.schedule(
            streaming.list_blocks()
        )

        self.shuffle: bool = shuffle
        self.buffer_size: Optional[int] = buffer_size

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        schedule: Schedule = self.schedules[0]

        if worker_info is not None:
            schedule = self.schedules[worker_info.id]

        streaming = SampleStreaming(
            self.dataset,
            tensors=self.tensors,
            use_local_cache=self.use_local_cache,
        )

        if self.shuffle:
            schedule.shuffle()

        stream = streaming.read(schedule)

        yield from TorchDataset.apply_transform(
            stream=TorchDataset.map_to_tensor(self.to_pytensor, stream),
            transform=self.transform,
        )

    def __len__(self):
        return sum(map(len, self.schedules))

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
            return tensor.astype(np.int32)
        if tensor.dtype == np.uint32:
            return tensor.astype(np.int64)
        if tensor.dtype == np.uint64:
            return tensor.astype(np.int64)

        return tensor.copy()

    def to_pytensor(self, tensor):
        return torch.from_numpy(self._remap_dtype(tensor))
