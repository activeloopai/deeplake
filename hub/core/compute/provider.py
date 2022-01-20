from abc import ABC, abstractmethod
from multiprocessing.managers import SyncManager

import ctypes
import time


class ComputeProvider(ABC):
    """An abstract base class for implementing a compute provider."""

    def __init__(self, workers):
        self.workers = workers

    def map_with_progressbar(self, func, iterable, total_length: int, desc: str = None):
        from tqdm.std import tqdm  # type: ignore
        import threading
        from threading import Thread

        progress_bar = tqdm(total=total_length, desc=desc)
        progress = self.manager().Value(ctypes.c_int64, 0)
        done = self.manager().Event()

        def sub_func(*args, **kwargs):
            class ProgUpdate:
                def __init__(self) -> None:
                    self.last_call = time.time()
                    self.acc = 0

                def pg_callback(self, done: int) -> None:
                    self.acc += done
                    if time.time() - self.last_call > 0.1:
                        progress.value += self.acc
                        self.acc = 0
                        self.last_call = time.time()

                def flush(self):
                    progress.value += self.acc

            pg = ProgUpdate()

            try:
                return func(pg.pg_callback, *args, **kwargs)
            finally:
                pg.flush()

        def update_pg(bar: tqdm, progress, done):
            prev = progress.value

            while prev < total_length:
                val = progress.value
                diff = val - prev

                if diff > 0:
                    bar.update(diff)
                prev = val

                time.sleep(0.1)

                if done.is_set():
                    break

        progress_thread: Thread = threading.Thread(
            target=update_pg, args=(progress_bar, progress, done)
        )
        progress_thread.start()

        try:
            result = self.map(sub_func, iterable)
        finally:
            done.set()
            progress_thread.join()
            progress_bar.close()

        return result

    @abstractmethod
    def manager(self) -> SyncManager:
        """Creates queue for specific provider"""

    @abstractmethod
    def map(self, func, iterable):
        """Applies 'func' to each element in 'iterable', collecting the results
        in a list that is returned.
        """

    @abstractmethod
    def close(self):
        """Closes the provider."""
