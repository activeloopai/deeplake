import os


def infer_schema(path):
    if not os.path.isdir(path):
        raise Exception('input path must be either a directory or file')

    schema = {}

    # TODO: determine input type

    # TODO: determine label type

    return schema
