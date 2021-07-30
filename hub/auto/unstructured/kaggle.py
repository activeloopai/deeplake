import hub
import glob
import os
from hub.util.exceptions import (
    ExternalCommandError,
    KaggleMissingCredentialsError,
    KaggleDatasetAlreadyDownloadedError,
)
from hub.constants import ENV_KAGGLE_KEY, ENV_KAGGLE_USERNAME


def _exec_command(command):
    out = os.system(command)
    if out != 0:
        raise ExternalCommandError(command, out)


def _set_environment_credentials_if_none(kaggle_credentials: dict = None):
    if kaggle_credentials is None:
        kaggle_credentials = {}
        if ENV_KAGGLE_USERNAME not in os.environ:
            username = kaggle_credentials.get("username", None)
            if not username:
                raise KaggleMissingCredentialsError(ENV_KAGGLE_USERNAME)
            os.environ[ENV_KAGGLE_USERNAME] = username
        if ENV_KAGGLE_KEY not in os.environ:
            key = kaggle_credentials.get("key", None)
            if not key:
                raise KaggleMissingCredentialsError(ENV_KAGGLE_KEY)
            os.environ[ENV_KAGGLE_KEY] = key


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
