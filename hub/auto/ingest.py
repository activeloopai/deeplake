from typing import Union
from hub.core.storage.provider import StorageProvider
from hub.auto.unstructured_dataset.image_classification import ImageClassification
import os


from hub import Dataset
from hub.util.kaggle import download_kaggle_dataset
from hub.util.exceptions import KaggleDatasetAlreadyDownloadedError

import warnings


def _dataset_has_tensors(**kwargs):
    ds = Dataset(**kwargs, mode="r")
    return len(ds.keys()) > 0


def _warn_kwargs(caller: str, **kwargs):
    if _dataset_has_tensors(**kwargs):
        warnings.warn("Dataset already exists, skipping ingestion and returning a read-only Dataset.")
        return  # no other warnings should print

    if "mode" in kwargs:
        warnings.warn("Argument `mode` should not be passed to `%s`. Ignoring and using `mode=\"write\"`." % caller)

    if "path" in kwargs:
        # TODO: generalize warns
        warnings.warn("Argument `path` should not be passed to `%s`. Ignoring and using `destination`." % caller)


# TODO: after plugins infra is available, move into `hub.Dataset`


def from_path(source: str, destination: Union[str, StorageProvider], **kwargs):
    """Copies unstructured data from `source` and structures/sends it to `destination`.

    Note:
        Be careful when providing sources to large datasets!
        This method copies data from `source` to `destination`.
        To be safe, you should assume the size of your dataset will consume 3-5x more space than expected.

    Args:
        source (str): Local-only path to where the unstructured dataset is stored.
        destination (str): Path to where the structured data will be stored.
        **kwargs: Args will be passed into `hub.Dataset`.

    Returns:
        A read-only `hub.Dataset` instance pointing to the structured data.
    """

    # TODO: test that ingestion methods ALWAYS return read-only datasets.

    _warn_kwargs("from_path", **kwargs)

    if isinstance(destination, StorageProvider):
        kwargs["storage"] = destination
    else:
        kwargs["path"] = destination

    # TODO: check for incomplete ingestion
    # TODO: try to resume progress for incomplete ingestion

    if _dataset_has_tensors(**kwargs):
        return Dataset(**kwargs, mode="r")

    ds = Dataset(**kwargs, mode="w")

    # TODO: auto detect which `UnstructuredDataset` subclass to use
    unstructured = ImageClassification(source)
    unstructured.structure(ds)

    # TODO: opt-in delete unstructured data after ingestion

    return Dataset(**kwargs, mode="r")


def from_kaggle(tag: str, source: str, destination: str, kaggle_credentials: dict={}, **kwargs):

    """Downloads the kaggle dataset with the given `tag` to this local machine, then that data is structured and copied into `destination`.

    Note:
        Be careful when providing tags to large datasets!
        This method downloads data from kaggle to the calling machine's local storage.
        To be safe, you should assume the size of the kaggle dataset being downloaded will consume 3-5x more space than expected.

    Args:
        tag (str): Kaggle dataset tag. Example: `"coloradokb/dandelionimages"` points to https://www.kaggle.com/coloradokb/dandelionimages
        source (str): Local-only path to where the unstructured kaggle dataset will be downloaded/unzipped.
        destination (str): Path to where the structured data will be stored.
        kaggle_credentials (dict): TODO:
        **kwargs: Args will be passed into `hub.Dataset`.

    Returns:
        A read-only `hub.Dataset` instance pointing to the structured data.
    """

    # TODO: make sure source and destination paths are not equal
    # TODO: make sure `destination` is an EMPTY directory
    # TODO: make sure `source` is a local path ONLY

    _warn_kwargs("from_kaggle", **kwargs)

    if _dataset_has_tensors(**kwargs):
        return Dataset(**kwargs, mode="r")

    try:
        download_kaggle_dataset(tag, local_path=source, kaggle_credentials=kaggle_credentials)
    except KaggleDatasetAlreadyDownloadedError as e:
        warnings.warn(e.message)

    ds = from_path(source=source, destination=destination, **kwargs)

    return ds