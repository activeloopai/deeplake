__all__ = ['get_parsers']

_parsers = []
_priorities = []


def get_parsers(priority_sort=True):
    if priority_sort:
        sorted_parsers = [x for _, x in sorted(zip(_priorities, _parsers))]
        return sorted_parsers
    return _parsers


def directory_parser(priority=0):
    def decorate(fn):
        _parsers.append(fn)
        _priorities.append(priority)
        return fn

    return decorate


@directory_parser(priority=0)
def image_classification(path):
    return None
