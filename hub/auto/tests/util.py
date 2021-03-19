import os
from pathlib import Path, PosixPath


def get_dataset_store(tag):
    """gets the absolute path to the test dataset named `tag`"""

    this_dir = Path(__file__).parent.absolute()
    return this_dir / "data" / tag


def _exec_command(command):
    print("using command", command)
    print("-----------------")
    out = os.system(command)
    print("-----------------")
    print("exit code:", out)
    assert out == 0


def _download_kaggle(tag, dataset_store):
    """deprecated -- not using kaggle datasets anymore"""

    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    assert username is not None
    assert key is not None
    print("got credentials %s@%s" % (username, key))
    print('downloading kaggle dataset "%s" to %s' % (tag, dataset_store))

    os.makedirs(dataset_store, exist_ok=True)
    setup = "cd %s && export KAGGLE_USERNAME=%s && export KAGLE_KEY=%s &&" % (
        dataset_store,
        username,
        key,
    )
    _exec_command("%s kaggle datasets download -d %s" % (setup, tag))
    _exec_command("%s unzip -n *.zip" % (setup))
