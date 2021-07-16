from collections import OrderedDict


class IterableOrderedDict(OrderedDict):
    def __iter__(self):
        for v in self.values():
            yield v
