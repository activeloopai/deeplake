from hub.auto.unstructured.image_classification import ImageClassification
from hub.util.exceptions import (
    ExternalCommandError,
    KaggleMissingCredentialsError,
    KaggleDatasetAlreadyDownloadedError,
)
import hub

import glob
import os

kaggle_credentials = {
    "username": "thisiseshan",
    "key": "lol",
}


_KAGGLE_USERNAME = "KAGGLE_USERNAME"
_KAGGLE_KEY = "KAGGLE_KEY"


def _exec_command(command):
    out = os.system(command)
    if out != 0:
        raise ExternalCommandError(command, out)


def _set_environment_credentials_if_none(kaggle_credentials: dict = {}):
    if _KAGGLE_USERNAME not in os.environ:
        username = kaggle_credentials.get("username", None)
        os.environ[_KAGGLE_USERNAME] = username
        if not username:
            raise KaggleMissingCredentialsError(_KAGGLE_USERNAME)
    if _KAGGLE_KEY not in os.environ:
        key = kaggle_credentials.get("key", None)
        os.environ[_KAGGLE_KEY] = key
        if not key:
            raise KaggleMissingCredentialsError(_KAGGLE_KEY)


def download_kaggle_dataset(tag: str, local_path: str, kaggle_credentials: dict = {}):
    """Calls the kaggle API (https://www.kaggle.com/docs/api) to download a kaggle dataset and unzip it's contents.

    Args:
        tag (str): Kaggle dataset tag. Example: `"coloradokb/dandelionimages"` points to https://www.kaggle.com/coloradokb/dandelionimages
        local_path (str): Path where the kaggle dataset will be downloaded and unzipped. Only local path downloading is supported.
        kaggle_credentials (dict): Credentials are gathered from the environment variables or `~/kaggle.json`.
            If those don't exist, the `kaggle_credentials` argument will be used.

    Raises:
        KaggleMissingCredentialsError: If no kaggle credentials are found.
        KaggleDatasetAlreadyDownloadedError: If the dataset `tag` already exists in `local_path`.
    """

    zip_files = glob.glob(os.path.join(local_path, "*.zip"))
    if len(zip_files) > 0:
        # TODO: this case means file did not finish unzipping (after unzip, it should be deleted)
        raise KaggleDatasetAlreadyDownloadedError(tag, local_path)
    subfolders = glob.glob(os.path.join(local_path, "*"))
    if len(subfolders) > 0:
        # TODO: this case means file finished unzipping and dataset is already there
        raise KaggleDatasetAlreadyDownloadedError(tag, local_path)

    _set_environment_credentials_if_none(kaggle_credentials)

    os.makedirs(local_path, exist_ok=True)
    setup = "cd %s &&" % (local_path)

    _exec_command("%s kaggle datasets download -d %s" % (setup, tag))
    _exec_command("%s unzip -n *.zip" % setup)
    _exec_command("%s rm *.zip" % setup)


tag = "olgabelitskaya/horse-breeds"
local = "./datasets/horse/source/1"
hub_path = "./datasets/horse/destination/1"


def run():
    download_kaggle_dataset(
        tag, local_path=local, kaggle_credentials=kaggle_credentials
    )
    ds = hub.Dataset(hub_path)
    unstructured = ImageClassification(source=local)
    unstructured.structure(ds, image_tensor_args={"sample_compression": "png"})


run()