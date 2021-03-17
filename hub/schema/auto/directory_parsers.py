import os
from tqdm import tqdm
import numpy as np
from glob import glob

from PIL import Image

import hub

__all__ = ['get_parsers']

_parsers = []
_priorities = []

USE_TQDM = True
IMAGE_EXTS = ['.jpg', '.png', '.jpeg']


def get_image_shape(path):
    img = Image.open(path)
    c = len(img.getbands())
    w, h = img.size
    return (c, h, w)


def get_children(path):
    """helper function to glob the given directory"""

    return glob(os.path.join(path, '*'))


def get_ext(path):
    ext = os.path.splitext(path)[-1]
    return ext.lower()


def files_are_of_extension(path, allowed_extensions):
    """helper function that checks if all files within the given directory have extensions that are allowed"""

    allowed_extensions = [ext.lower() for ext in allowed_extensions]
    children = get_children(path)
    return all([get_ext(child) in allowed_extensions for child in children])


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
    children = get_children(path)

    # check if all children are directories
    if not all([os.path.isdir(child) for child in children]):
        return None

    # check if children's contents are all image files
    for child in children:
        if not files_are_of_extension(child, IMAGE_EXTS):
            return None

    # parse dataset
    data = {}
    max_shape = np.zeros(3, dtype=np.uint8)  # CHW
    for child in tqdm(children,
                      desc='parsing image classification dataset',
                      total=len(children)):
        label = os.path.basename(child)

        if label in data.keys():
            raise Exception('label %s is a duplicate' % label)

        filepaths = get_children(child)
        filenames = []
        for filepath in filepaths:
            shape = np.array(get_image_shape(filepath))
            max_shape = np.maximum(max_shape, shape)
            filenames.append(os.path.basename(filepath))

        data[label] = {
            'filenames': filenames,
            'max_shape': max_shape,
            'dir': child
        }

    # create schema
    max_shape = tuple([int(x) for x in max_shape])
    schema = {
        'image':
        hub.schema.Image(shape=(None, None, None),
                         dtype='uint8',
                         max_shape=max_shape),
        'label':
        hub.schema.ClassLabel(num_classes=len(children))
    }

    return schema, data
