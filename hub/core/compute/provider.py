from abc import ABC, abstractmethod
import time


class SharedValue(ABC):
    @abstractmethod
    def set(self, value):
        """Set shared value"""

    @abstractmethod
    def get(self) -> int:
        """Get shared value"""


class ComputeProvider(ABC):
    """An abstract base class for implementing a compute provider."""

    def __init__(self, workers):
        self.workers = workers

    def map_with_progressbar(self, func, iterable, total_length: int, desc: str = None):
        from tqdm.std import tqdm  # type: ignore
        import threading
        from threading import Thread

        progress_bar = tqdm(total=total_length, desc=desc)
        progress = self.create_shared_value()
        done = self.create_shared_value()

        def sub_func(*args, **kwargs):
            class ProgUpdate:
                def __init__(self) -> None:
                    self.last_call = time.time()
                    self.acc = 0

                def pg_callback(self, done: int) -> None:
                    self.acc += done
                    if time.time() - self.last_call > 0.1:
                        progress.set(progress.get() + self.acc)
                        self.acc = 0
                        self.last_call = time.time()

                def flush(self):
                    progress.set(progress.get() + self.acc)

            pg = ProgUpdate()

            try:
                return func(pg.pg_callback, *args, **kwargs)
            finally:
                pg.flush()

        def update_pg(bar: tqdm, progress, done):
            prev = progress.get()

            while prev < total_length:
                val = progress.get()
                diff = val - prev

                if diff > 0:
                    bar.update(diff)
                prev = val

                time.sleep(0.1)

                if done.get() > 0:
                    break

        progress_thread: Thread = threading.Thread(
            target=update_pg, args=(progress_bar, progress, done)
        )
        progress_thread.start()

        try:
            result = self.map(sub_func, iterable)
        finally:
            done.set(1)
            progress_thread.join()
            progress_bar.close()

        return result

    @abstractmethod
    def create_shared_value(self) -> SharedValue:
        """Creates value that can be shared between threads"""

    @abstractmethod
    def map(self, func, iterable):
        """Applies 'func' to each element in 'iterable', collecting the results
        in a list that is returned.
        """

    @abstractmethod
    def close(self):
        """Closes the provider."""
