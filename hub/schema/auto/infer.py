import os
from glob import glob

__all__ = ['infer_schema_and_shape']


def _find_root(path):
    """
    find the root of the dataset within the given path.
    the "root" is defined as being the path to a subdirectory within path that has > 1 folder/file (if applicable).

    in other words, if there is a directory structure like:
    dataset -
        Images -
            class1 -
                img.jpg
                ...
            class2 -
                img.jpg
                ...
            ...

    the output of this function should be "dataset/Images/" as that is the root.
    """

    subs = glob(os.path.join(path, '*'))
    if len(subs) > 1:
        return path
    return _find_root(subs[0])


def infer_schema_and_shape(path):
    if not os.path.isdir(path):
        raise Exception('input path must be either a directory or file')

    root = _find_root(path)

    schema = {}
    shape = (1, )  # TODO

    # TODO: determine input type
    # does it match an image classification dataset?

    # TODO: determine label type

    return schema, shape
