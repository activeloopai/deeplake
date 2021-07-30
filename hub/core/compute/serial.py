from hub.core.compute.provider import ComputeProvider


class SerialProvider(ComputeProvider):
    def __init__(self, workers):
        self.workers = workers

    def map(self, func, iterable):
        return map(func, iterable)
