from hub.core.compute.provider import ComputeProvider


class SerialProvider(ComputeProvider):
    def __init__(self):
        pass

    def map(self, func, iterable):
        return list(map(func, iterable))

    def map_with_progressbar(self, func, iterable, total_length: int):
        from tqdm.std import tqdm  # type: ignore

        result = list()

        for x in tqdm(iterable, total=total_length):
            result.append(func(x))

        return result

    def create_queue(self):
        raise NotImplementedError("no queues in serial provider")

    def close(self):
        return
