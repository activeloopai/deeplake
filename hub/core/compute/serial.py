from hub.core.compute.provider import ComputeProvider


class SerialProvider(ComputeProvider):
    def __init__(self):
        pass

    def map(self, func, iterable):
        return map(func, iterable)
