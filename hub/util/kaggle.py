from hub.util.exceptions import MissingKaggleCredentialsError
from hub.core.storage.local import LocalProvider
import os


_KAGGLE_USERNAME = "KAGGLE_USERNAME"
_KAGGLE_KEY = "KAGGLE_KEY"


def _exec_command(command):
    # TODO: remove prints
    print("using command", command)
    print("-----------------")
    out = os.system(command)
    print("-----------------")
    print("exit code:", out)
    assert out == 0  # TODO: remove assert


def _set_environment_credentials(credentials: dict={}):
    if _KAGGLE_USERNAME not in os.environ:
        os.environ[_KAGGLE_USERNAME] = credentials.get("username", None)
    if _KAGGLE_KEY not in os.environ:
        os.environ[_KAGGLE_KEY] = credentials.get("key", None)


def _get_kaggle_username(credentials: dict={}):
    username = os.environ.get(_KAGGLE_USERNAME, None)
    if not username:
        raise MissingKaggleCredentialsError(_KAGGLE_USERNAME)
    return username


def _get_kaggle_key(credentials: dict={}):
    key = os.environ.get(_KAGGLE_KEY, None)
    if not key:
        raise MissingKaggleCredentialsError(_KAGGLE_KEY)
    return key


def download_kaggle(tag: str, local_path: str, credentials: dict={}):
    # TODO: docstring

    if os.path.isdir(local_path):
        return  # TODO: 

    _set_environment_credentials(credentials)
    username = _get_kaggle_username(credentials)
    key = _get_kaggle_key(credentials)

    os.makedirs(local_path, exist_ok=True)
    setup = "cd %s && export KAGGLE_USERNAME=%s && export KAGLE_KEY=%s &&" % (
        local_path,
        username,
        key,
    )

    _exec_command("%s kaggle datasets download -d %s" % (setup, tag))
    _exec_command("%s unzip -n *.zip" % setup)
    _exec_command("%s rm *.zip" % setup)