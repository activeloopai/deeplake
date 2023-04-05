from collections import OrderedDict


class IterableOrderedDict(OrderedDict):
    def __iter__(self):
        yield from self.values()
