from hub.auto.converter import Converter
import os


from hub import Dataset
from hub.util.kaggle import download_kaggle_dataset
from hub.util.exceptions import KaggleDatasetAlreadyDownloadedError

import warnings


DEFAULT_LOCAL_PATH = "./datasets/"

HUB_DATASET_SUBDIR = "hub"
UNSTRUCTURED_DATASET_SUBDIR = "unstructured"


def _warn_kwargs(**kwargs):
    if Dataset(**kwargs).exists():
        warnings.warn("Dataset already exists, skipping ingestion and returning a read-only Dataset.")
        return  # no other warnings should print

    if "mode" in kwargs:
        warnings.warn("Mode should not be passed to `Dataset.from_path`. Using write mode.")


def from_path(unstructured_path: str, **kwargs):
    """Creates a hub dataset from unstructured data.

    Note:
        This copies the data into hub format.
        Be careful when using this with large datasets.

    Args:
        path (str): Path to the data to be converted

    Returns:
        A Dataset instance whose path points to the hub formatted
        copy of the data.
    """

    _warn_kwargs(**kwargs)

    ds = Dataset(**kwargs, mode="w")
    if ds.exists():
        return Dataset(**kwargs, mode="r")

    converter = Converter(unstructured_path)
    converter.from_image_classification(ds)

    # TODO: opt-in delete unstructured data after ingestion

    # TODO: return in read-only mode
    return ds


def from_kaggle(tag: str, path: str=None, local_path: str=DEFAULT_LOCAL_PATH, kaggle_credentials: dict={}, **kwargs):
    # TODO: docstring
    # TODO: make sure path and local path are not equal
    # TODO: make path variable names more obvious

    """Downloads a kaggle dataset and creates a new hub dataset. Calls `from_path` on the kaggle dataset.

    Note:
        This downloads data from kaggle and then copies it into hub, be careful when providing tags to large datasets.

    Args:
        tag (str): Kaggle dataset tag. Example: `"coloradokb/dandelionimages"` points to https://www.kaggle.com/coloradokb/dandelionimages
        path (str): Path to where the hub dataset will be uploaded. Passed into `hub.Dataset(path=path)`. See `hub.Dataset` for more information.
            If not provided, `path` will be set to `DEFAULT_LOCAL_PATH`/`tag`/`HUB_DATASET_SUBDIR`.
        local_path (str): Local path to a directory where the kaggle dataset will be downloaded/unzipped.
            If not provided, `local_path` will be set to `DEFAULT_LOCAL_PATH`/`tag`/`HUB_DATASET_SUBDIR`.
        **kwargs: Args will be passed into `Dataset`.

    Returns:
        A new `hub.Dataset` that is created at `path`.
    """

    kaggle_download_path = os.path.join(local_path, tag, UNSTRUCTURED_DATASET_SUBDIR)
    if not path:
        path = os.path.join(local_path, tag, HUB_DATASET_SUBDIR)
        # TODO: warning?

    _warn_kwargs(path=path, **kwargs)

    read_only_ds = Dataset(path=path, **kwargs, mode="r")
    if read_only_ds.exists():
        return read_only_ds

    # TODO: try reading hub dataset before downloading (in case the unstructured folder was deleted, but the hub folder remains)

    try:
        download_kaggle_dataset(tag, local_path=kaggle_download_path, kaggle_credentials=kaggle_credentials)
    except KaggleDatasetAlreadyDownloadedError as e:
        warnings.warn(e.message)

    ds = from_path(kaggle_download_path, path=path, **kwargs)

    return ds