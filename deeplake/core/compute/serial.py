from deeplake.core.compute.provider import ComputeProvider


class SerialProvider(ComputeProvider):
    def __init__(self):
        pass

    def map(self, func, iterable):
        return list(map(func, iterable))

    def map_with_progressbar(self, func, iterable, total_length: int, desc=None):
        progress_bar = self.get_progressbar(total_length, desc)

        def sub_func(*args, **kwargs):
            def pg_callback(value: int):
                progress_bar.update(value)

            return func(pg_callback, *args, **kwargs)

        result = self.map(sub_func, iterable)

        return result

    def create_queue(self):
        raise NotImplementedError("no queues in serial provider")

    def close(self):
        return
