from abc import ABC, abstractmethod
from typing import Optional
import threading
import warnings
from tqdm.std import tqdm  # type: ignore
from tqdm import TqdmWarning  # type: ignore


def get_progress_bar(total_length, desc):
    warnings.simplefilter("ignore", TqdmWarning)

    return tqdm(
        total=total_length,
        desc=desc,
        bar_format="{desc}: {percentage:.0f}%|{bar}| {n:.0f}/{total_fmt} [{elapsed}<{remaining}",
    )


def get_progress_thread(progress_bar, progress_queue):
    def update_pg(bar, queue):
        while True:
            r = queue.get()
            if r is not None:
                bar.update(r)
            else:
                break

    progress_thread = threading.Thread(
        target=update_pg, args=(progress_bar, progress_queue)
    )
    progress_thread.start()
    return progress_thread


class ComputeProvider(ABC):
    """An abstract base class for implementing a compute provider."""

    def __init__(self, workers):
        self.workers = workers

    def map_with_progress_bar(
        self,
        func,
        iterable,
        total_length: int,
        desc: Optional[str] = None,
        pbar=None,
        pqueue=None,
    ):
        progress_bar = pbar or get_progress_bar(total_length, desc)
        progress_queue = pqueue or self.create_queue()

        def sub_func(*args, **kwargs):
            def pg_callback(value: int):
                progress_queue.put(value)  # type: ignore[trust]

            return func(pg_callback, *args, **kwargs)

        progress_thread = get_progress_thread(progress_bar, progress_queue)

        try:
            result = self.map(sub_func, iterable)
        finally:
            progress_queue.put(None)  # type: ignore[trust]
            if pqueue is None and hasattr(progress_queue, "close"):
                progress_queue.close()
            progress_thread.join()
            if pbar is None:
                progress_bar.close()

        return result

    @abstractmethod
    def create_queue(self):
        """Creates queue for specific provider"""

    @abstractmethod
    def map(self, func, iterable):
        """Applies 'func' to each element in 'iterable', collecting the results
        in a list that is returned.
        """

    @abstractmethod
    def close(self):
        """Closes the provider."""
