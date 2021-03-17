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
    return (h, w, c)


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
def image_classification(path, scheduler='single', workers=1):
    children = get_children(path)

    # check if all children are directories
    if not all([os.path.isdir(child) for child in children]):
        return None

    # check if children's contents are all image files
    for child in children:
        if not files_are_of_extension(child, IMAGE_EXTS):
            return None

    # parse dataset
    data = []
    class_names = set()
    max_shape = np.zeros(3, dtype=np.uint8)  # CHW
    for child in tqdm(children,
                      desc='parsing image classification dataset',
                      total=len(children)):
        label = os.path.basename(child)

        filepaths = get_children(child)
        for filepath in filepaths:
            shape = np.array(get_image_shape(filepath))
            max_shape = np.maximum(max_shape, shape)
            data.append((filepath, label.lower()))
            class_names.add(label.lower())

    class_names = list(sorted(list(class_names)))

    # create schema
    max_shape = tuple([int(x) for x in max_shape])
    schema = {
        'image':
        hub.schema.Image(shape=(None, None, None),
                         dtype='uint8',
                         max_shape=max_shape),
        'label':
        hub.schema.ClassLabel(names=class_names)
    }

    # create transform for putting data into hub format
    @hub.transform(schema=schema, scheduler=scheduler, workers=workers)
    def upload_data(sample):
        img = Image.open(sample[0])
        img = np.asarray(img)
        label = class_names.index(sample[1])
        return {'image': img, 'label': label}

    return upload_data(data)
