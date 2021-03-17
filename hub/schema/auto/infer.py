import os
from glob import glob
from hub.schema.auto.directory_parsers import get_parsers

__all__ = ['infer_schema_and_shape']

_directory_parsers = get_parsers()


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

    # blank schema by default
    schema = None
    shape = (1, )  # TODO

    # go through all functions created using the `directory_parser` decorator in
    # `hub.schema.auto.directory_parsers`
    for parser in _directory_parsers:
        schema = parser(root)
        if schema is not None:
            break

    # TODO: determine input type
    # does it match an image classification dataset?

    # TODO: determine label type

    if schema is None:
        raise Exception(
            'could not infer schema for the root "%s". either add a new parser to'
            % root +
            '`hub.schema.auto.directory_parsers` or write a custom schema.')

    return schema, shape
