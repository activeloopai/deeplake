from hub.core.compute.provider import ComputeProvider, SharedValue


class SerialProvider(ComputeProvider):
    def __init__(self):
        pass

    def map(self, func, iterable):
        return list(map(func, iterable))

    def map_with_progressbar(self, func, iterable, total_length: int, desc=None):
        from tqdm.std import tqdm  # type: ignore

        pbar = tqdm(total=total_length, desc=desc)

        def sub_func(*args, **kwargs):
            def pg_callback(value: int):
                pbar.update(value)

            return func(pg_callback, *args, **kwargs)

        result = self.map(sub_func, iterable)

        return result

    def create_shared_value(self) -> SharedValue:
        return ManagedValue()

    def close(self):
        return


class ManagedValue(SharedValue):
    def __init__(self) -> None:
        super().__init__()
        self.val = 0

    def set(self, val) -> None:
        self.val = val

    def get(self):
        return self.val
