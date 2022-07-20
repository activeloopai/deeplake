from typing import Iterable, Optional, Sequence, List, Union
from hub.constants import MB
from hub.integrations.pytorch.common import PytorchTransformFunction

from hub.util.iterable_ordered_dict import IterableOrderedDict
from hub.core.io import (
    DistributedScheduler,
    SampleStreaming,
    Schedule,
    SequentialMultithreadScheduler,
    ShufflingSchedulerWrapper,
    SingleThreadScheduler,
    MultiThreadedNaiveScheduler,
)
from hub.integrations.pytorch.shuffle_buffer import ShuffleBuffer

import torch
import torch.utils.data
import torch.distributed as dist

from torch.multiprocessing import Queue, Process
from torch._utils import ExceptionWrapper
from torch.utils.data.dataloader import DataLoader
from torch.utils.data._utils.worker import ManagerWatchdog
from torch._C import _remove_worker_pids, _set_worker_pids, _set_worker_signal_handlers
from torch.utils.data._utils.signal_handling import _set_SIGCHLD_handler
from warnings import warn
from queue import Empty

import numpy as np
import hub


mp = torch.multiprocessing.get_context()


def identity(x):
    return x


def use_scheduler(num_workers: int, ensure_order: bool):
    if num_workers <= 1:
        return SingleThreadScheduler()
    else:
        if ensure_order:
            return MultiThreadedNaiveScheduler(num_workers)
        else:
            return SequentialMultithreadScheduler(num_workers)


def cast_type(tensor: np.ndarray):
    if tensor.dtype == np.uint16:
        return tensor.astype(np.int32)
    if tensor.dtype == np.uint32:
        return tensor.astype(np.int64)
    if tensor.dtype == np.uint64:
        return tensor.astype(np.int64)

    return tensor


def copy_tensor(x):
    try:
        return cast_type(x.copy())
    except AttributeError:
        return bytes(x)


def _process(tensor, transform: PytorchTransformFunction):
    tensor = IterableOrderedDict((k, copy_tensor(tensor[k])) for k in tensor)
    tensor = transform(tensor)
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
    Indicates worker done with its schedule
    """

    def __init__(self) -> None:
        pass


def _worker_loop(
    wid: int,
    dataset,
    tensors,
    tobytes,
    use_local_cache: bool,
    schedule: Schedule,
    transform: PytorchTransformFunction,
    request_queue: Queue,
    data_queue: Queue,
    workers_done,
):
    torch.set_num_threads(1)
    _set_worker_signal_handlers()

    try:
        # let's don't reinvent the wheel here and use pytorch Watchdog
        watchdog = ManagerWatchdog()

        streaming = SampleStreaming(
            dataset,
            tensors=tensors,
            tobytes=tobytes,
            use_local_cache=use_local_cache,
        )

        schedule.shuffle()

        it = iter(streaming.read(schedule))  # data samples iterator for current worker
        iteration_end: bool = False
        requested: int = 0  # indicate value of samples requested from worker

        while watchdog.is_alive():

            # publish requested data
            if not iteration_end and requested > 0:
                try:
                    data = next(it)

                    data = _process(data, transform)
                    data = {k: torch.as_tensor(v) for k, v in data.items()}

                    data_queue.put((wid, data))
                    requested -= 1
                    del data

                except Exception as e:
                    if isinstance(e, StopIteration):
                        data_queue.put((wid, _StopIteration()))
                        iteration_end = True
                        requested = 0
                        continue

                    # propagate exception back to parent
                    data_queue.put(
                        (wid, ExceptionWrapper(where=f"shuffle worker thread-{wid}"))
                    )
                    iteration_end = True
                    requested = 0
                    continue

            # process commands from parent
            r: Union[_ContinueIteration, None]
            try:
                # do only quick check if more data requested
                if requested > 0:
                    r = request_queue.get_nowait()
                else:
                    r = request_queue.get(timeout=0.05)
            except Empty:
                # TODO can do some prefetch at this point
                # however, blocking thread for long might make it unresponsive.

                # Just continue for now
                continue

            if r is None:
                break

            if workers_done.is_set() or iteration_end:
                iteration_end = True
                continue

            # Continue Iteration instruction
            if isinstance(r, _ContinueIteration):
                requested += r.n

    except KeyboardInterrupt:
        pass

    # gracefully close connections
    if workers_done.is_set():
        data_queue.cancel_join_thread()
        data_queue.close()


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
        self.prefetch = 64

        self.workers: List[Process] = []
        self.request_queues: List[Queue] = [Queue() for _ in range(self.num_workers)]
        self.data_queue: Queue = Queue()
        self.queue_size = [0 for _ in range(self.num_workers)]
        self.active_workers = [False for _ in range(self.num_workers)]
        self.workers_done = mp.Event()
        self.shutdown = False
        self._worker_pids_set = False

        for i in range(self.num_workers):
            input_queue: Queue = self.request_queues[i]
            input_queue.cancel_join_thread()

            w = mp.Process(
                target=_worker_loop,
                args=(
                    i,
                    dataset.dataset,
                    dataset.tensors,
                    dataset.tobytes,
                    dataset.use_local_cache,
                    dataset.schedules[i],
                    dataset.transform,
                    input_queue,
                    self.data_queue,
                    self.workers_done,
                ),
            )
            w.daemon = True
            w.start()

            self.workers.append(w)
            self.active_workers[i] = True
            self.request_queues[i].put_nowait(_ContinueIteration(self.prefetch))

        _set_worker_pids(id(self), tuple(w.pid for w in self.workers))  # type: ignore[misc]
        _set_SIGCHLD_handler()
        self._worker_pids_set = True

    def __iter__(self):
        shuffle_buffer = (
            ShuffleBuffer(self.buffer_size) if self.buffer_size > 0 else None
        )

        while any(self.active_workers):
            try:
                wid, data = self.data_queue.get(
                    timeout=hub.constants.PYTORCH_DATALOADER_TIMEOUT
                )

                if isinstance(data, ExceptionWrapper):
                    data.reraise()

                # Handling emergency termination of sequence
                if isinstance(data, _StopIteration):
                    self.active_workers[wid] = False
                    continue

                # else return sample
                self.queue_size[wid] += 1

                # exchange sample with shuffle buffer
                if shuffle_buffer is not None:
                    data = shuffle_buffer.exchange(data)

                    if data is not None:
                        yield data
                else:
                    yield data

                # if low on messages in a queue, request more
                if self.queue_size[wid] > (self.prefetch * 0.25):
                    self.request_queues[wid].put(
                        _ContinueIteration(self.queue_size[wid])
                    )
                    self.queue_size[wid] = 0
            except Empty:
                for i, worker in enumerate(self.workers):
                    if self.active_workers[i] and not worker.is_alive():
                        self.active_workers[i] = False
                        warn(f"worker id-{i} terminated unexpectedly")

        self._shutdown_all()

        # clear all samples from shuffle buffer
        if shuffle_buffer is not None:
            while not shuffle_buffer.emtpy():
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
                    self.request_queues[wid].put(None)
                    self.workers[wid].join(
                        timeout=hub.constants.PYTORCH_DATALOADER_TIMEOUT
                    )

                for queue in self.request_queues:
                    queue.cancel_join_thread()
                    queue.close()

                self.data_queue.cancel_join_thread()
                self.data_queue.close()
            finally:
                if self._worker_pids_set:
                    _remove_worker_pids(id(self))
                    self._worker_pids_set = False

                for worker in self.workers:
                    if worker.is_alive():
                        worker.terminate()

            self.shutdown = True


class ShufflingIterableDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        dataset,
        use_local_cache: bool = False,
        tensors: Sequence[str] = None,
        tobytes: Union[bool, Sequence[str]] = False,
        transform: PytorchTransformFunction = PytorchTransformFunction(),
        num_workers: int = 1,
        buffer_size: int = 0,
    ) -> None:

        super().__init__()

        assert num_workers >= 1

        self.dataset = dataset
        self.num_workers = num_workers
        self.transform = transform
        self.tensors = tensors
        self.tobytes = tobytes
        self.use_local_cache = use_local_cache

        if dist.is_initialized():
            self.scheduler = ShufflingSchedulerWrapper(
                DistributedScheduler(num_workers)
            )
        else:
            self.scheduler = ShufflingSchedulerWrapper(
                MultiThreadedNaiveScheduler(self.num_workers)
            )

        self.scheduler = ShufflingSchedulerWrapper(self.scheduler)
        streaming = SampleStreaming(
            dataset,
            tensors=self.tensors,  # type: ignore
            tobytes=self.tobytes,
            use_local_cache=use_local_cache,
        )

        self.schedules: List[Schedule] = self.scheduler.schedule(
            streaming.list_blocks()
        )

        self.buffer_size: int = buffer_size * MB

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
        tensors: Sequence[str] = None,
        tobytes: Union[bool, Sequence[str]] = False,
        transform: PytorchTransformFunction = PytorchTransformFunction(),
        num_workers: int = 1,
        shuffle: bool = False,
        buffer_size: int = 0,
        return_index: bool = True,
        pad_tensors: bool = False,
    ) -> None:
        super().__init__()

        self.dataset = dataset
        self.transform = transform
        self.tensors = tensors
        self.tobytes = tobytes
        self.pad_tensors = pad_tensors

        self.use_local_cache = use_local_cache
        self.scheduler = use_scheduler(num_workers, shuffle)

        if dist.is_initialized():
            self.scheduler = DistributedScheduler(num_workers)

        if shuffle:
            self.scheduler = ShufflingSchedulerWrapper(self.scheduler)

        streaming = SampleStreaming(
            dataset,
            tensors=self.tensors,  # type: ignore
            tobytes=self.tobytes,
            use_local_cache=use_local_cache,
            pad_tensors=self.pad_tensors,
        )

        self.schedules: List[Schedule] = self.scheduler.schedule(
            streaming.list_blocks()
        )

        self.shuffle: bool = shuffle
        self.buffer_size: int = buffer_size * MB
        self.return_index: bool = return_index

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        schedule: Schedule = self.schedules[0]

        if worker_info is not None:
            schedule = self.schedules[worker_info.id]

        streaming = SampleStreaming(
            self.dataset,
            tensors=self.tensors,
            tobytes=self.tobytes,
            use_local_cache=self.use_local_cache,
            return_index=self.return_index,
            pad_tensors=self.pad_tensors,
        )

        if self.shuffle:
            schedule.shuffle()

        stream = streaming.read(schedule)

        for data in stream:
            yield _process(data, self.transform)

    def __len__(self):
        return sum(map(len, self.schedules))


class SubIterableDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        dataset,
        use_local_cache: bool = False,
        tensors: Optional[Sequence[str]] = None,
        tobytes: Union[bool, Sequence[str]] = False,
        transform: PytorchTransformFunction = PytorchTransformFunction(),
        num_workers: int = 1,
        buffer_size: int = 512,
        batch_size: int = 1,
        return_index: bool = True,
        pad_tensors: bool = False,
    ) -> None:
        super().__init__()

        self.torch_datset = TorchDataset(
            dataset,
            use_local_cache,
            tensors,
            tobytes,
            transform,
            num_workers=num_workers,
            shuffle=True,
            return_index=return_index,
            pad_tensors=pad_tensors,
        )

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.buffer_size = buffer_size * MB

        if self.buffer_size == 0:
            warn("setting buffer_size = 0 will result in poor shuffling distribution")

    def __iter__(self):
        buffer = ShuffleBuffer(self.buffer_size) if self.buffer_size > 0 else None

        sub_loader = DataLoader(
            self.torch_datset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=identity,
        )

        it = iter(sub_loader)

        try:
            while True:
                next_batch = next(it)
                for val in next_batch:
                    if buffer is not None:
                        result = buffer.exchange(val)
                        if result:
                            yield result
                    else:
                        yield val

                del next_batch

        except StopIteration:
            pass

        if buffer is not None:
            while not buffer.emtpy():
                yield buffer.exchange(None)

        del sub_loader

    def __len__(self):
        return len(self.torch_datset)
