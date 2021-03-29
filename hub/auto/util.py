import os
from glob import glob
import numpy as np
from tqdm import tqdm

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
    if get_ext(path) not in IMAGE_EXTS:
        return None

    img = Image.open(path)
    c = len(img.getbands())
    w, h = img.size
    return (h, w, c)


def infer_shape(path, p=1.0, use_tqdm=True):
    """infers the shape of the given path. if path is a directory, it will analyze all subdirectories & their image files"""


    # TODO: handle stochastic inference (determine shape based on a small random subset of the data)
    assert p <= 1 and p > 0

    # TODO: if path is file (not directory), handle

    children = get_children(path, only_dirs=False)
    np.random.shuffle(children)
    num_samples = int(float(p) * float(len(children)))
    children = children[:num_samples]
    print(num_samples)
    print(len(children))

    max_shape = np.zeros(3, dtype=np.uint8)  # CHW
    all_same_shape = True
    image_shape = None  # if shape is all the same, use that instead of max_shape
    for child in tqdm(
        children,
        desc="parsing image classification dataset",
        total=len(children),
        disable=not use_tqdm,
    ):
        # if child is a file, handle
        if os.path.isfile(child):
            shape = get_image_shape(child)

            if shape is None:
                continue

            shape = np.array(shape)

            # check if all images have the same shape
            if all_same_shape:
                if image_shape is None:
                    image_shape = shape
                elif not np.array_equal(image_shape, shape):
                    all_same_shape = False

            max_shape = np.maximum(max_shape, shape)

        # if child is a directory, handle
        elif os.path.isdir(child):
            raise NotImplementedError()

        else:
            raise Exception('%s is not a child or file.' % str(child))

        """
        label = os.path.basename(child)

        filepaths = get_children(child)
        for filepath in filepaths:
            # ignore non-image extension files
            if util.get_ext(filepath) not in util.IMAGE_EXTS:
                continue

            shape = np.array(util.get_image_shape(filepath))

            # check if all images have the same shape
            if all_same_shape:
                if image_shape is None:
                    image_shape = shape
                elif not np.array_equal(image_shape, shape):
                    all_same_shape = False

            max_shape = np.maximum(max_shape, shape)
            data.append((filepath, label.lower()))
            class_names.add(label.lower())
        """

    if all_same_shape:
        return shape, None

    return (None, None, None), max_shape


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
