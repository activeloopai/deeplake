from deeplake.core.compute.provider import ComputeProvider, get_progress_bar


class SerialProvider(ComputeProvider):
    def __init__(self):
        pass

    def map(self, func, iterable):
        return list(map(func, iterable))

    def map_with_progress_bar(
        self,
        func,
        iterable,
        total_length: int,
        desc=None,
        pbar=None,
        pqueue=None,
    ):
        progress_bar = pbar or get_progress_bar(total_length, desc)

        def sub_func(*args, **kwargs):
            def pg_callback(value: int):
                progress_bar.update(value)

            return func(pg_callback, *args, **kwargs)

        result = self.map(sub_func, iterable)

        return result

    def create_queue(self):
        return None

    def close(self):
        return
