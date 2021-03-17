import os
from glob import glob
from PIL import Image

IMAGE_EXTS = ['.jpg', '.png', '.jpeg']

_parsers = []
_priorities = []


def get_parsers(priority_sort=True):
    if priority_sort:
        sorted_parsers = [x for _, x in sorted(zip(_priorities, _parsers))]
        return sorted_parsers
    return _parsers


def directory_parser(priority=0):
    """
    a directory parser function is a function that takes in a path & returns a dataset.
    these functions make it easier to extend the dataset inference domain. 
    functions should be as general as possible.

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


def get_children(path):
    """helper function to glob the given directory"""

    return glob(os.path.join(path, '*'))


def get_image_shape(path):
    img = Image.open(path)
    c = len(img.getbands())
    w, h = img.size
    return (h, w, c)


def get_ext(path):
    ext = os.path.splitext(path)[-1]
    return ext.lower()


def files_are_of_extension(path, allowed_extensions):
    """
    helper function that checks if any files within the given directory have extensions
    that are allowed
    """

    allowed_extensions = [ext.lower() for ext in allowed_extensions]
    children = get_children(path)
    # print(set([get_ext(child) for child in children]))
    # print(set([child for child in children if get_ext(child) == '']))
    return any([get_ext(child) in allowed_extensions for child in children])
