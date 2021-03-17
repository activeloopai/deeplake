__all__ = ['get_parsers']

_parsers = []
_priorities = []


def get_parsers(priority_sort=True):
    if priority_sort:
        sorted_parsers = [x for _, x in sorted(zip(_priorities, _parsers))]
        return sorted_parsers
    return _parsers


def directory_parser(priority=0):
    """
    a directory parser function is a function that takes in a path & returns a schema. 
    these functions make it easier to extend the schema infer domain. functions should
    be as general as possible.

    Parameters
    ----------
    priority: int
        an arbitrary number that the parsers will be sorted by 
        (lower the number = higher the priority)
    """
    def decorate(fn):
        _parsers.append(fn)
        _priorities.append(priority)
        return fn

    return decorate


@directory_parser(priority=0)
def image_classification(path):
    return None
