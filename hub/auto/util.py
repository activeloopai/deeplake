import os
from glob import glob

from PIL import Image

IGNORE_EXTS = [".DS_Store"]
IMAGE_EXTS = [".jpg", ".png", ".jpeg"]
CSV_EXTS = [".csv"]


class DirectoryParserState:
    def __init__(self):
        self._parsers = []
        self._priorities = []

    def get_parsers(self, priority_sort=True):
        if priority_sort:
            sorted_parsers = [
                x for _, x in sorted(zip(self._priorities, self._parsers))
            ]
            return sorted_parsers
        return self._parsers

    def directory_parser(self, priority=0):
        """
        a directory parser function is a function that takes in a path & returns a
        dataset. these functions make it easier to extend the dataset inference domain.
        functions should be as general as possible.

        Parameters
        ----------
        priority: int
            an arbitrary number that the parsers will be sorted by
            (lower the number = higher the priority)
        """

        def decorate(fn):
            self._parsers.append(fn)
            self._priorities.append(priority)
            return fn

        return decorate


def should_be_ignored(path):
    for ignore in IGNORE_EXTS:
        if path.endswith(ignore):
            return True
    return False


def get_children(path, only_dirs=False):
    """helper function to glob the given directory"""

    children = glob(os.path.join(path, "*"))
    children = [child for child in children if not should_be_ignored(child)]
    if only_dirs:
        children = [child for child in children if os.path.isdir(child)]
    return children


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
