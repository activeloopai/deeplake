import os
from glob import glob

import hub_v1
from hub_v1.auto import util

state = util.DirectoryParserState()

__all__ = ["infer_dataset"]


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

    subs = glob(os.path.join(path, "*"))
    hub_dir = os.path.join(path, "hub")
    if hub_dir in subs:
        subs.remove(hub_dir)  # ignore the hub directory
    subs = [
        sub for sub in subs if os.path.isdir(sub)
    ]  # only keep directories (ignore files)
    if len(subs) == 1:
        return _find_root(subs[0])
    return path


def infer_dataset(path, scheduler="single", workers=1):
    # TODO: handle s3 path

    if not os.path.isdir(path):
        raise Exception("input path must be either a directory")

    hub_path = os.path.join("./", path, "hub")

    if os.path.isdir(hub_path):
        print('inferred dataset found in "%s", using that' % hub_path)
        return hub_v1.Dataset(hub_path, mode="r")

    root = _find_root(path)
    ds = None

    directory_parsers = state.get_parsers()
    if len(directory_parsers) <= 0:
        raise Exception("directory parsers list was empty.")

    # go through all functions created using the `directory_parser` decorator in
    # `hub_v1.schema.auto.directory_parsers`
    for parser in directory_parsers:
        ds = parser(root, scheduler, workers)
        if ds is not None:
            break

    if ds is None:
        raise Exception(
            'could not infer dataset for the root "%s". either add a new parser to'
            % root
            + "`hub_v1.schema.auto.directory_parsers` or write a custom transform + schema."
        )

    ds.store(hub_path)  # TODO: handle s3
    return hub_v1.Dataset(hub_path, mode="r")
