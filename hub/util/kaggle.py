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


def download_kaggle(tag: str, local_path: str):
    # TODO: docstring

    if os.path.isdir(local_path):
        return  # TODO: 

    username = os.environ.get(_KAGGLE_USERNAME, None)
    if not username:
        raise MissingKaggleCredentialsError(_KAGGLE_USERNAME)
    key = os.environ.get(_KAGGLE_KEY, None)
    if not key:
        raise MissingKaggleCredentialsError(_KAGGLE_KEY)

    # TODO: remove prints
    print("got credentials %s@%s" % (username, key))
    print('downloading kaggle dataset "%s" to %s' % (tag, local_path))

    os.makedirs(local_path, exist_ok=True)
    setup = "cd %s && export KAGGLE_USERNAME=%s && export KAGLE_KEY=%s &&" % (
        local_path,
        username,
        key,
    )

    _exec_command("%s kaggle datasets download -d %s" % (setup, tag))
    _exec_command("%s unzip -n *.zip" % setup)
    _exec_command("%s rm *.zip" % setup)