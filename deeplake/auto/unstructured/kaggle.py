from threading import local
from typing import Optional
import deeplake
import glob
import os
from deeplake.util.exceptions import (
    ExternalCommandError,
    KaggleMissingCredentialsError,
    KaggleDatasetAlreadyDownloadedError,
)
from deeplake.constants import ENV_KAGGLE_KEY, ENV_KAGGLE_USERNAME
from zipfile import ZipFile
from typing import Optional


def _exec_command(command):
    out = os.system(command)
    if out != 0:
        raise ExternalCommandError(command, out)


def _set_environment_credentials_if_none(kaggle_credentials: Optional[dict] = None):
    if kaggle_credentials is not None:
        username = kaggle_credentials.get("username", None)
        if not username:
            raise KaggleMissingCredentialsError(ENV_KAGGLE_USERNAME)
        os.environ[ENV_KAGGLE_USERNAME] = username
        key = kaggle_credentials.get("key", None)
        if not key:
            raise KaggleMissingCredentialsError(ENV_KAGGLE_KEY)
        os.environ[ENV_KAGGLE_KEY] = key
    else:
        if ENV_KAGGLE_USERNAME not in os.environ:
            raise KaggleMissingCredentialsError(ENV_KAGGLE_USERNAME)
        if ENV_KAGGLE_KEY not in os.environ:
            raise KaggleMissingCredentialsError(ENV_KAGGLE_KEY)


def download_kaggle_dataset(
    tag: str,
    local_path: str,
    kaggle_credentials: Optional[dict] = None,
    exist_ok: bool = False,
):
    """Calls the kaggle API (https://www.kaggle.com/docs/api) to download a kaggle dataset and unzip it's contents.

    Args:
        tag (str): Kaggle dataset tag. Example: `"coloradokb/dandelionimages"` points to https://www.kaggle.com/coloradokb/dandelionimages
        local_path (str): Path where the kaggle dataset will be downloaded and unzipped. Only local path downloading is supported.
        kaggle_credentials (dict): Credentials are gathered from the environment variables or `~/kaggle.json`.
            If those don't exist, the `kaggle_credentials` argument will be used.
        exist_ok (bool): If the kaggle dataset was already downloaded, and `exist_ok` is True, no error is thrown.

    Raises:
        KaggleMissingCredentialsError: If no kaggle credentials are found.
        KaggleDatasetAlreadyDownloadedError: If the dataset `tag` already exists in `local_path`.
    """

    zip_files = glob.glob(os.path.join(local_path, "*.zip"))
    subfolders = glob.glob(os.path.join(local_path, "*"))

    if len(zip_files) > 0:
        # TODO: this case means file did not finish unzipping (after unzip, it should be deleted)

        if not exist_ok:
            raise KaggleDatasetAlreadyDownloadedError(tag, local_path)

        print(
            f'Kaggle dataset "{tag}" was already downloaded, but not extracted. Extracting...'
        )

    elif len(subfolders) > 0:
        # TODO: this case means file finished unzipping and dataset is already there

        if not exist_ok:
            raise KaggleDatasetAlreadyDownloadedError(tag, local_path)

        print(f'Kaggle dataset "{tag}" was already downloaded and extracted.')
        return

    _set_environment_credentials_if_none(kaggle_credentials)

    os.makedirs(local_path, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(local_path)

    _exec_command("kaggle datasets download -d %s" % (tag))

    for item in os.listdir():
        if item.endswith(".zip"):
            file_name = os.path.abspath(item)
            zip_ref = ZipFile(file_name)
            zip_ref.extractall(local_path)
            zip_ref.close()
            os.remove(file_name)

    os.chdir(cwd)
