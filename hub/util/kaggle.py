import os
import glob

from hub.util.exceptions import MissingKaggleCredentialsError, KaggleDatasetAlreadyDownloadedError
from hub.core.storage.local import LocalProvider


_KAGGLE_USERNAME = "KAGGLE_USERNAME"
_KAGGLE_KEY = "KAGGLE_KEY"


def _exec_command(command):
    out = os.system(command)
    assert out == 0  # TODO: replace assert with Exception


def _set_environment_credentials_if_none(credentials: dict={}):
    if _KAGGLE_USERNAME not in os.environ:
        username = credentials.get("username", None)
        os.environ[_KAGGLE_USERNAME] = username
        if not username:
            raise MissingKaggleCredentialsError(_KAGGLE_USERNAME)
    if _KAGGLE_KEY not in os.environ:
        key = credentials.get("key", None)
        os.environ[_KAGGLE_KEY] = key
        if not key:
            raise MissingKaggleCredentialsError(_KAGGLE_KEY)


def download_kaggle(tag: str, local_path: str, credentials: dict={}):
    # TODO: docstring

    zip_files = glob.glob(os.path.join(local_path, "*.zip"))
    if len(zip_files) > 0:
        raise KaggleDatasetAlreadyDownloadedError(tag, local_path)

    _set_environment_credentials_if_none(credentials)

    os.makedirs(local_path, exist_ok=True)
    setup = "cd %s &&" % (local_path)

    _exec_command("%s kaggle datasets download -d %s" % (setup, tag))
    _exec_command("%s unzip -n *.zip" % setup)
    _exec_command("%s rm *.zip" % setup)